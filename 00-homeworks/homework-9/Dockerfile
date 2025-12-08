FROM agrigorev/model-2025-hairstyle:v1

# Copy Lambda handler into /var/task
COPY lambda_function_optimised.py /var/task/

# Install required packages
RUN pip install onnx onnxruntime pillow numpy

# Set handler environment
ENV LAMBDA_HANDLER=lambda_function_optimised.handler

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/aws-lambda-rie", "python3", "-m", "awslambdaric"]
CMD ["lambda_function_optimised.handler"]
