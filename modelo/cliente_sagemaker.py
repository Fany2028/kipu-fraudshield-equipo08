import json
import os

import boto3

ENDPOINT = os.getenv(
    "KIPU_SAGEMAKER_ENDPOINT",
    "kipu-fraudshield-ep3-endpoint-v3"
)

REGION = os.getenv(
    "AWS_DEFAULT_REGION",
    "us-east-1"
)

def predecir_sagemaker(transaccion: dict):
    cliente = boto3.client(
        "sagemaker-runtime",
        region_name=REGION
    )

    respuesta = cliente.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(transaccion).encode("utf-8")
    )

    resultado = json.loads(
        respuesta["Body"].read().decode("utf-8")
    )

    return resultado