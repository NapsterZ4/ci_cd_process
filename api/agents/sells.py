from langchain_core.tools import tool

@tool
def buscar_precio(producto: str) -> dict:
    """

    :param producto:
    :return:
    """
    catalogo = {
        "laptop": 1200,
        "monitor": 350,
        "teclado": 75,
        "mouse": 25,
        "webcam": 90
    }
    p = producto.lower().strip()
    if p in catalogo:
        return {
            "message": f"{p}: ${catalogo[p]}"
        }

    return {
        "message": f"Producto '{p}' no encontrado. Disponibles: {', '.join(catalogo.keys())}",
    }

@tool
def aplicar_descuento(precio: float, porcentaje: float) -> dict:
    """

    :param precio:
    :param porcentaje:
    :return:
    """
    final = precio * (1 - porcentaje / 100)
    return {
        "message": f"Precio original: ${precio:.2f} → Con {porcentaje}% dto: ${final:.2f}"
    }
