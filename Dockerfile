# Usa uma imagem base oficial do Python
FROM python:3.9

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de requisitos para dentro do container
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o restante do seu projeto para o diretório de trabalho
COPY . .

# Cria a pasta Models caso ela não exista (precaução)
# e garante permissões de leitura/escrita para evitar erros de permissão
RUN mkdir -p Models && chmod 777 Models

# O Hugging Face Spaces roda na porta 7860 por padrão.
# Precisamos rodar o uvicorn apontando para essa porta.
# Como seu arquivo está na pasta API, o comando é API.apiprojeto:app
CMD ["uvicorn", "API.apiprojeto:app", "--host", "0.0.0.0", "--port", "7860"]