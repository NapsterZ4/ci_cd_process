from typing import Literal, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from api.agents.sells import buscar_precio, aplicar_descuento
from api.agents.support import estado_pedido, iniciar_devolucion
from api.agents.coding import coding_generator
from config import OPENAI_API_KEY
from api.schemas.prediction import Estado


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    api_key=OPENAI_API_KEY,
)

agente_ventas = create_react_agent(
    llm,
    tools=[buscar_precio, aplicar_descuento],
    prompt="Eres un agente de ventas. Ayudas con precios,"
           " productos y descuentos. Sé conciso.",
)

agente_codigo = create_react_agent(
    llm,
    tools=[coding_generator],
    prompt="Eres un generador de codigos en la consola de linux para ejecutar en linux"
)

agente_soporte = create_react_agent(
    llm,
    tools=[estado_pedido, iniciar_devolucion],
    prompt=(
        "Eres un agente de soporte al cliente especializado en gestión de pedidos y devoluciones. "
        "Tu objetivo es resolver la solicitud del cliente en la menor cantidad de pasos posible, "
        "usando las herramientas disponibles de forma proactiva sin pedir confirmaciones innecesarias. "
        "Cuando el cliente pregunte por un pedido, consulta el estado inmediatamente. "
        "Cuando solicite una devolución, inicia el proceso directamente extrayendo el ID del pedido "
        "y el motivo de su mensaje. Si falta información crítica como el ID del pedido, pídesela "
        "de forma breve y directa. Responde siempre en español, Se creativo en tu respuesta y que sea larga"
    ),
)

# ============================================================
# ESTADO Y GRAFO CON ROUTER
# ============================================================

def router(state: Estado) -> Estado:
    ultimo_msg = state["messages"][-1].content if state["messages"] else ""

    if isinstance(ultimo_msg, list):
        ultimo_msg = " ".join(
            bloque.get("text", "") if isinstance(bloque, dict) else str(bloque)
            for bloque in ultimo_msg
        )

    respuesta = llm.invoke([
        SystemMessage(content=(
            "Clasifica la intención del usuario en UNA sola palabra:\n"
            "- 'ventas' si pregunta por precios, productos, catálogo o descuentos\n"
            "- 'soporte' si pregunta por pedidos, envíos, devoluciones o problemas\n"
            "- 'codigo' si pregunta por generar codigo\n"
            "Responde SOLO con la palabra."
        )),
        HumanMessage(content=ultimo_msg),
    ])

    destino = respuesta.content.strip().lower()
    if destino not in ("ventas", "soporte", "codigo"):
        destino = "ventas"

    print(f" Router → {destino}")
    return {**state, "siguiente": destino}


def nodo_ventas(state: Estado) -> Estado:
    resultado = agente_ventas.invoke({"messages": state["messages"]})
    return {**state, "messages": resultado["messages"]}


def nodo_soporte(state: Estado) -> Estado:
    resultado = agente_soporte.invoke({"messages": state["messages"]})
    return {**state, "messages": resultado["messages"]}


def nodo_codigo(state: Estado) -> Estado:
    resultado = agente_codigo.invoke({"messages": state["messages"]})
    return {**state, "messages": resultado["messages"]}


def decidir_ruta(state: Estado) -> Literal["ventas", "soporte"]:
    return state["siguiente"]


grafo = StateGraph(Estado)
grafo.add_node("router", router)
grafo.add_node("ventas", nodo_ventas)
grafo.add_node("soporte", nodo_soporte)
grafo.add_node("codigo", nodo_codigo)
grafo.add_edge(START, "router")
grafo.add_conditional_edges("router", decidir_ruta, {"ventas": "ventas", "soporte": "soporte", "codigo": "codigo"})
grafo.add_edge("ventas", END)
grafo.add_edge("soporte", END)
grafo.add_edge("codigo", END)
app = grafo.compile()
