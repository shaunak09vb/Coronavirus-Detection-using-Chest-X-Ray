FROM python:3.8
WORKDIR /main
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8080
COPY . /main
CMD streamlit run --server.port 8080 --server.enableCORS false main.py