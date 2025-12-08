FROM agrigorev/model-2025-hairstyle:v1

COPY lambda_function.py .

RUN pip install onnx onnxruntime pillow torchvision numpy

ENV LAMBDA_HANDLER=lambda_function.handler

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/aws-lambda-rie", "python3", "-m", "awslambdaric"]
CMD ["lambda_function.handler"]

