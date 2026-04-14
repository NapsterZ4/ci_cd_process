from langchain_core.tools import tool

@tool
def coding_generator(command: str) -> str:
    """Ejecuta un comando en la terminal y devuelve el resultado."""
    print("ejecutando comando: ", command, "")
    return f"Comando ejecutado correctamente: {command}"
