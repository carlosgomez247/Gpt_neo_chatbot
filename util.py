import re
def procesar_entrada(texto):
    texto = texto.lower()
    texto = re.sub(r'\W', ' ', texto)    
    texto = re.sub(r'\s+', ' ', texto, flags=re.I)
    texto = re.sub('[áàäâ]', 'a', texto)
    texto = re.sub('[éèëê]', 'e', texto)
    texto = re.sub('[íìïî]', 'i', texto)
    texto = re.sub('[óòöô]', 'o', texto)
    texto = re.sub('[úùüû]', 'u', texto)
    return texto