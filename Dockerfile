FROM python:3.6

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /model_health_matrix

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

# Set the entry-point as gunicorn
ENTRYPOINT ["gunicorn"]

# Set the default command to run (which can be overridden)
CMD ["-c","gunicorn_config.py","-b","0.0.0.0:80", "run:app"]

