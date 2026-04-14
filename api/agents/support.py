from langchain_core.tools import tool

@tool
def estado_pedido(pedido_id: str) -> str:
    """

    :param pedido_id:
    :return:
    """
    pedidos = {
        "PED-001": "Enviado - llega el 15 de abril",
        "PED-002": "En preparación",
        "PED-003": "Entregado el 10 de abril",
    }
    return pedidos.get(
        pedido_id.upper(),
        f"Pedido '{pedido_id}' no encontrado."
    )


@tool
def iniciar_devolucion(pedido_id: str, motivo: str) -> str:
    """

    :param pedido_id:
    :param motivo:
    :return:
    """
    return (f"Devolución iniciada para {pedido_id.upper()}."
            f" Motivo: {motivo}. Recibirás un email con la etiqueta de envío.")
