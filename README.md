# github-analyzer

Este pequeño proyecto lo he creado para investigar y aprender como funciona langchan y langgraph. Después de una profunda conversación con Chatgpt ha surgido esta idea, un analizador de proyectos de github, la lógica es sencilla, le pasas un proyecto y el resto es todo seguir las instrucciones. 

Tienes varias opciones para analizar/preguntar/pdfs sobre el proyecto, como no langraph se encarga de orquestar todo esto con una lista de posibles casos. En la siguiente imagen se aprecian las opciones:



## Requisitos

- **Python 3.10+** (probado con 3.12)
- **git** en el `PATH`
- **OpenCode** instalado y con un proveedor/modelo configurado (`opencode`).
  El LLM se sirve localmente con `opencode serve`.
- (Opcional) **Semgrep** para el análisis estático. Sin él, la app usa un
  escáner heurístico propio.

### Instalación

```bash
# 1. Entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Dependencias principales
pip install -r requirements.txt

# 3. (Recomendado) embeddings semánticos + Semgrep
pip install -r requirements-optional.txt
```

> `requirements-optional.txt` instala `sentence-transformers` (para embeddings
> semánticos reales) y `semgrep`. Son opcionales: sin ellos la aplicación
> funciona con un fallback de hashing y un escáner heurístico, pero con peor
> calidad de búsqueda y de detección.

### Arrancar el servidor de modelos

En otra terminal:

```bash
opencode serve
```

Queda escuchando en `http://127.0.0.1:4096`. La aplicación descubre
automáticamente el modelo por defecto consultando `/config/providers`.

### Configuración (opcional)

```bash
cp .env.example .env
```

Variables principales:

| Variable | Descripción |
|---|---|
| `OPENCODE_BASE_URL` | URL del servidor (por defecto `http://127.0.0.1:4096`) |
| `OPENCODE_PROVIDER_ID` / `OPENCODE_MODEL_ID` | Modelo concreto; si se dejan vacíos se usa el modelo por defecto |
| `OPENCODE_AGENT` | Agente de OpenCode (opcional: `build`, `plan`, ...) |
| `OPENCODE_SERVER_USERNAME` / `OPENCODE_SERVER_PASSWORD` | Autenticación básica opcional |
| `EMBEDDINGS_PROVIDER` | `auto` \| `huggingface` \| `hash` |
| `HF_MODEL_NAME` | Modelo de sentence-transformers para embeddings |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Tamaño/solape de los chunks |
| `RETRIEVAL_K` | Nº de fragmentos recuperados por pregunta |

---

## Uso

```bash
python main.py
```

```text
================================
       AI CODE ANALYZER
================================

Introduce la URL del repositorio:
> https://github.com/usuario/proyecto

[1] Analizar repositorio
[2] Hacer preguntas
[3] Análisis completo
[4] Generar informe PDF
[5] Salir
```

1. **Analizar repositorio**: clona, filtra, trocea e indexa el código.
2. **Hacer preguntas**: chat RAG sobre el código indexado (`salir` para volver).
3. **Análisis completo**: ejecuta Semgrep y el LLM explica los hallazgos.
4. **Generar informe PDF**: crea `reports/<proyecto>-analysis.pdf`.
5. **Salir**.

> En consolas Windows antiguas, si los caracteres `✓` o las tildes se ven mal,
> ejecuta primero `chcp 65001`.

---

## Arquitectura por capas

El proyecto separa deliberadamente las responsabilidades:

```text
Python        → operaciones normales (git clone, leer archivos, Semgrep)
LangGraph     → controla el FLUJO: nodos + transiciones + estado
LangChain     → piezas de LLM/RAG: prompts, retriever, vector store, cadenas
OpenCode Server→ proporciona acceso al modelo (API HTTP local)
LLM           → comprende, razona y genera respuestas
```

### Los tres grafos de LangGraph

| Grafo | Flujo | Cuándo se ejecuta |
|---|---|---|
| Indexación | `clone → read → chunk → index` | Opción 1 |
| Preguntas | `retrieve → answer` | Cada pregunta (opción 2) |
| Análisis | `semgrep → explain` | Opción 3 |
| Informe | `report` | Opción 4 |

Cada **nodo** es una función Python normal que recibe el estado y devuelve la
parte que actualiza. El **estado** (`AnalyzerState`) es el diccionario tipado
que viaja de nodo en nodo. Para la primera versión usamos flujos fijos
(controlados); un agente autónomo que decide qué herramienta usar sería el
siguiente paso.

### Cómo se conecta el LLM (OpenCode Server)

OpenCode Server **no** expone una API compatible con OpenAI. Usa su propia API
por sesiones. Por eso implementamos `OpenCodeLLM`, que hereda de
`BaseChatModel` (la clase de LangChain para modelos de chat) y traduce:

```text
LangChain:  [SystemMessage, HumanMessage] ──► OpenCode: POST /session/{id}/message
                                               { system: "...",
                                                 parts: [{type:"text", text:"..."}],
                                                 model: {providerID, modelID} }
```

El resto de la app solo ve un `llm` estándar de LangChain: puede cambiarse por
cualquier otro modelo sin tocar el pipeline.

### Cómo funciona el RAG

En lugar de enviar todo el repositorio en cada pregunta:

1. **Embeddings**: cada chunk de código se convierte en un vector numérico.
2. **Vector store**: los vectores se guardan (`InMemoryVectorStore`, sin
   servidor externo; se puede sustituir por FAISS/Chroma).
3. **Retriever**: `vectorstore.as_retriever(...)` busca los `k` chunks más
   parecidos a la pregunta (similitud coseno).
4. **Prompt + LLM**: solo ese contexto + la pregunta llegan al modelo.

Las fuentes se muestran al usuario para que la respuesta sea verificable.

---
