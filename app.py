import os
import json
from flask import Flask, jsonify, request
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from sqlalchemy import text
from typing import TypedDict, Annotated, List, Any
import operator
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

# Importar herramientas
from tools.registrar_solicitud import create_tool_registrar_solicitud
from tools.consultar_estado import create_tool_consultar_estado
from tools.actualizar_solicitud import create_tool_actualizar_solicitud
from tools.busqueda_documental import create_tool_busqueda_documental

# Cargar variables de entorno
#load_dotenv() 

# --- CONFIGURACIÓN DE FASTAPI Y VARIABLES DE ENTORNO ---
app = Flask(__name__)

# Configuración: Estas variables serán inyectadas por Cloud Run
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
POSTGRES_URI = os.environ.get("POSTGRES_URI")
ELASTIC_URL = os.environ.get("ELASTIC_URL")
ELASTIC_USER = os.environ.get("ELASTIC_USER")
ELASTIC_PASSWORD = os.environ.get("ELASTIC_PASSWORD")
ELASTIC_INDEX = os.environ.get("ELASTIC_INDEX")

# Configuración: Langsmith
os.environ["LANGSMITH_ENDPOINT"] = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.environ.get("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "S09-eiagurp")

# --- INICIALIZACIÓN DE COMPONENTES GLOBALES (Se ejecuta una sola vez al inicio del servidor) ---

# 1. Conexión a Base de Datos SQL
try:
    DB_CONN = SQLDatabase.from_uri(POSTGRES_URI)
    print("Conexión a PostgreSQL establecida.")
except Exception as e:
    print(f"ERROR: Fallo al conectar con la base de datos SQL. Herramientas SQL deshabilitadas. Detalle: {e}")
    DB_CONN = None

# 2. Configuración del LLM
MODEL = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# 3. Configuración de Memoria Persistente (Checkpointer)
try:
    # Usamos ConnectionPool para gestionar las conexiones a la DB para el Checkpointer
    CONN_POOL = ConnectionPool(
        conninfo=POSTGRES_URI,
        min_size=1,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    MEMORY_SAVER = PostgresSaver(CONN_POOL)
    print("Checkpointer (PostgresSaver) inicializado.")
except Exception as e:
    print(f"ERROR: Fallo al inicializar PostgresSaver. La memoria no será persistente. Detalle: {e}")
    # Fallback
    from langgraph.checkpoint.memory import MemorySaver as FallbackMemorySaver
    MEMORY_SAVER = FallbackMemorySaver()

# 4. Vector Store
def setup_vector_store():
    """Configura la conexión al vector store."""
    if not all([ELASTIC_URL, ELASTIC_PASSWORD, ELASTIC_INDEX]):
        print("ADVERTENCIA: Faltan credenciales/URL de Elasticsearch.")
        return None
    try:
        embedding = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
        vector_store = ElasticsearchStore(
            es_url=ELASTIC_URL,
            es_user=ELASTIC_USER,
            es_password=ELASTIC_PASSWORD,
            index_name=ELASTIC_INDEX,
            embedding=embedding
        )
        print("Conexión a Elasticsearch establecida.")
        return vector_store
    except Exception as e:
        print(f"ERROR: Fallo al conectar con Elasticsearch. Herramienta de documentación deshabilitada. Detalle: {e}")
        return None

VECTOR_STORE = setup_vector_store()

# 4. Creación de todas las herramientas
TOOLS = []
if DB_CONN:
    TOOLS.append(create_tool_registrar_solicitud(DB_CONN))
    TOOLS.append(create_tool_consultar_estado(DB_CONN))
    TOOLS.append(create_tool_actualizar_solicitud(DB_CONN))
if VECTOR_STORE:
    TOOLS.append(create_tool_busqueda_documental(VECTOR_STORE))
print(f"Total de herramientas disponibles: {len(TOOLS)}")

# --- DEFINICIÓN DEL GRAFO Y AGENTES ---

# Estado del grafo
class AgenteState(TypedDict):
    messages: Annotated[List[Any], operator.add]
    next: str

# Nodo de agente
def agent_node(state, agent_instance):
    """Ejecuta el agente."""
    result = agent_instance.invoke(state)
    return {"messages": [result["messages"][-1]]}

# Nodo supervisor
def supervisor_node(state: AgenteState):
    """Decide qué agente debe ser el siguiente en responder (DOCUMENTACION o USUARIO_EXTERNO)."""
    user_query = state["messages"][-1].content
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Eres un agente supervisor de un sistema de reembolsos. Tu tarea es enrutar la consulta del usuario. "
        "--- REGLA DE PRIORIDAD CRÍTICA --- "
         "1. Si el historial muestra que el AI pidió un dato obligatorio (ej. nombre, monto, fecha) y la última consulta del usuario es una respuesta corta que proporciona ese dato, **DEBES** asumir que es una continuación de la acción y enrutar a **USUARIO_EXTERNO**."
         "2. Solo enruta a **DOCUMENTACION** si la pregunta del usuario es una pregunta teórica nueva (ej. '¿Qué documentos necesito...?' o '¿Qué dice la política...?')."
         "Basado en la última consulta, decide a qué equipo debe ir: "
         "- **DOCUMENTACION**: Si la pregunta es sobre políticas, procedimientos, requisitos, qué documentos llevar, o cualquier información teórica general. "
         "- **USUARIO_EXTERNO**: Si la pregunta implica una ACCIÓN sobre una solicitud: registrar una solicitud, consultar el estado o actualizar una solicitud. "
         "Tu respuesta DEBE ser una de las siguientes palabras ÚNICAMENTE: DOCUMENTACION, USUARIO_EXTERNO."
        ),
        ("human", f"Última consulta del usuario: {user_query}")
    ])
    
    cadena_decision = decision_prompt | MODEL
    decision = cadena_decision.invoke({"user_query": user_query}).content.strip().upper()
    
    # Lógica de enrutamiento
    if "DOCUMENTACION" in decision:
        return {"next": "documentacion"}
    elif "USUARIO_EXTERNO" in decision:
        return {"next": "usuario_externo"}
    else:
        # Falla segura al agente de acción si la decisión no es clara
        return {"next": "usuario_externo"} 

# Función para crear el agente de Usuario basado en el rol de la solicitud
def create_agent_for_role(rol: str, usuario_logueado_user: str, usuario_logueado_fullname: str):
    """Crea el agente de LangGraph con el conjunto de herramientas apropiado según el rol."""
    
    if rol == "Administrador":
        # Filtra herramientas para el rol de Admin
        toolkit = [t for t in TOOLS if t.name in ['registrar_solicitud', 'consultar_estado', 'actualizar_solicitud']]
        prompt_instruccion = (
            f"El usuario logueado es **{usuario_logueado_user}** y tiene acceso total. "
            "Cuando uses la herramienta 'consultar_estado_tool', **NO incluyas el argumento 'usuario'** en la llamada. "
            f"Para la herramienta 'registrar_solicitud_tool', utiliza **{usuario_logueado_user}** y **{usuario_logueado_fullname}** automáticamente para los argumentos: 'usuario' y 'nombre_asegurado', respectivamente."
        )
    elif rol == "General":
        # Filtra herramientas para el rol General
        toolkit = [t for t in TOOLS if t.name in ['registrar_solicitud', 'consultar_estado']]
        prompt_instruccion = (
            f"El usuario logueado es **{usuario_logueado_user}**. "
            f"Para las herramientas 'registrar_solicitud_tool' y 'consultar_estado_tool', utiliza **{usuario_logueado_user}** y **{usuario_logueado_fullname}** automáticamente para los argumentos: 'usuario' y 'nombre_asegurado', respectivamente. "
            "Solo puedes consultar solicitudes asociadas a tu usuario."
        )
    else:
        toolkit = [] 
        prompt_instruccion = "Rol desconocido. No tienes permisos para realizar acciones."

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
          f"Eres el agente de soporte para reembolsos médicos. {prompt_instruccion} "
          "Si el usuario menciona un 'Beneficiario', úsalo para el argumento 'nombre_beneficiario'. "
          "Céntrate siempre en responder únicamente a la última pregunta del usuario. NO repitas ni resumas acciones o confirmaciones de solicitudes ya procesadas en turnos anteriores de la conversación si es que el usuario no te las pide."
          "Usa las herramientas disponibles solo cuando sea necesario y sé cortés."),
        ("human", "{messages}")
    ])

    agent_instance = create_react_agent(MODEL, toolkit, checkpointer=MEMORY_SAVER, prompt=prompt)
    return agent_instance

# Función para crear el agente de Documentación (solo tiene acceso a busqueda_documental)
def create_documentacion_agent():
    toolkit = [t for t in TOOLS if t.name == 'busqueda_documental'] 
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Eres el agente de documentación. Tu ÚNICA función es usar la herramienta 'busqueda_documental' para encontrar el procedimiento o pasos de reembolsos en la base de datos vectorial y resumir la información encontrada. "
         "Si la herramienta no devuelve información, indica al usuario que no encontraste ese detalle."
        ),
        ("human", "{messages}")
    ])
    
    agent_instance = create_react_agent(MODEL, toolkit, checkpointer=MEMORY_SAVER, prompt=prompt)
    return agent_instance

# Construcción del grafo
def build_agent_graph(agente_usuario, agente_documentacion):
    """Construye el grafo supervisor que conecta los agentes de rol."""
    workflow = StateGraph(AgenteState)
    
    # Nodos
    workflow.add_node("documentacion", lambda state: agent_node(state, agente_documentacion))
    workflow.add_node("usuario_externo", lambda state: agent_node(state, agente_usuario))
    workflow.add_node("supervisor", supervisor_node)
    
    # Flujo
    workflow.set_entry_point("supervisor")
    
    workflow.add_edge("documentacion", END)
    workflow.add_edge("usuario_externo", END)
    
    workflow.add_conditional_edges(
        "supervisor", 
        lambda x: x["next"],
        {
            "documentacion": "documentacion",
            "usuario_externo": "usuario_externo",
            END: END
        }
    )
    
    app_instance = workflow.compile(checkpointer=MEMORY_SAVER)
    return app_instance

# Agente de Documentación (es global porque no depende de datos del usuario)
AGENTE_DOCUMENTACION = create_documentacion_agent()

# --- RUTA API PRINCIPAL ---

@app.route('/agent', methods=['GET', 'POST'])
def handle_agent_request():
    """
    Endpoint principal para interactuar con el agente de LangChain. 
    Es invocado por el frontend de Next.js.
    """
    
    # 1. Capturar parámetros de la solicitud
    if request.method == 'GET':
        data = request.args
    elif request.method == 'POST':
        try:
            data = request.get_json()
        except Exception:
            data = request.args
            
    # Parámetros necesarios para el agente (enviados desde el Front-end)
    session_id = data.get('id_agente')
    user_input = data.get('msg')
    user_role = data.get('user_role', 'General')
    username = data.get('username', 'usuario_default')
    display_name = data.get('display_name', 'Usuario Desconocido')

    # Validación mínima
    if not all([session_id, user_input, user_role, username, display_name]):
        return jsonify({
            "response": "Error: Faltan parámetros de sesión, mensaje o usuario.",
            "status": "error"
        }), 400

    # 2. Crear el agente de Usuario basado en el rol de la solicitud
    agente_usuario_externo = create_agent_for_role(
        user_role, 
        username, 
        display_name
    )
    
    # 3. Construir el Grafo con los agentes correctos
    agent_app = build_agent_graph(agente_usuario_externo, AGENTE_DOCUMENTACION)

    # 4. Preparar la invocación
    config = {"configurable": {"thread_id": session_id}}
    langchain_messages = [HumanMessage(content=user_input)]
    
    try:
        # 5. Invocar al agente
        response = agent_app.invoke(
            {"messages": langchain_messages},
            config=config
        )
        
        output = response["messages"][-1].content
        
        return jsonify({
            "response": output,
            "thread_id": session_id,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error al ejecutar el agente de LangGraph: {e}")
        return jsonify({
            "response": f"Ocurrió un error interno al ejecutar el agente.",
            "status": "error",
            "error_detail": str(e)
        }), 500

# Endpoint de verificación de salud
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "service": "Agente de Reembolsos Médicos API"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)