## create Dockerfile

```python
FROM python:3.8-slim-buster
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]
```

```python
docker build -t boston .
docker run -p 8501:8501 boston  
```

## Upload it to docker hub

```docker
docker tag 91febe5e77d0 imsumitkumar/boston:latest
docker push imsumitkumar/boston:latest
```

## Create Dockerrun.aws.json


```jsx
{
    "AWSEBDockerrunVersion": "1",
    "Image": {
      "Name": "user_name/image_name",
      "Update": "true"
    },
    "Ports": [
      {
        "ContainerPort": 8501,
        "HostPort": 8501
      }
    ],
    "Logging": "/var/log/nginx"
  }
```

## New app in Aws BeanStalk

```
name it
choose docker
choose upload and upload Dockerrun.aws.json
more configration
    - instance = choose general purpose and 8 gb
    - capacity 
    - save
Create app
```