FROM python:3.9.17-bullseye

COPY requirements.txt /

EXPOSE 5004

RUN pip3 install -r /requirements.txt

COPY . /app

WORKDIR /app

ENTRYPOINT ["./gunicorn.sh"]



