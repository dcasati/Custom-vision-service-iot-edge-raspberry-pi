FROM python:3.6

COPY /build/amd64-requirements.txt amd64-requirements.txt

ADD app /app

RUN pip install -r amd64-requirements.txt
RUN pip3 install -U tensorflow==1.9.0rc2


# Expose the port
EXPOSE 80

# Set the working directory
WORKDIR /app

# Run the flask server for the endpoints
CMD python app.py
