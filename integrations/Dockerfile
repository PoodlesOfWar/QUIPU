FROM python:3.12-slim
RUN useradd --system --uid 10001 app
WORKDIR /app
COPY server.py ./
COPY site ./site
USER 10001
EXPOSE 8090
HEALTHCHECK --interval=20s --timeout=4s CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/health',timeout=3).status==200 else 1)"
CMD ["python", "server.py"]
