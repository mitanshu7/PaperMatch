FROM python:3.13-slim

EXPOSE 8001

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY backend/ .

CMD ["uvicorn", "main:app",  "--host", "0.0.0.0", "--port", "8001"]