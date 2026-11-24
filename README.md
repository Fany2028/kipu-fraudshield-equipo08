# Kipu FraudShield - Equipo 08

Proyecto academico de deteccion de posibles transacciones fraudulentas.
Caso 3 | EP2 y EP3

## Componentes

- Limpieza y preparacion de datos mediante Python.
- Entrenamiento de un modelo Random Forest.
- API REST desarrollada con FastAPI.
- Autenticacion por API Key.
- Limitacion de solicitudes con SlowAPI.
- Inferencia local y mediante AWS SageMaker.
- Pruebas automatizadas con pytest.

## Requisitos

- Python y pip.
- Acceso a los artefactos del modelo local.
- Credenciales AWS validas y permisos para invocar SageMaker, si se utiliza esa funcionalidad.

## Instalacion en Windows (PowerShell)

Desde la carpeta principal del proyecto:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Archivos del modelo local

La API actual necesita estos archivos dentro de la carpeta modelo:

- kipu_fraudshield_ep3_conservador.joblib
- kipu_fraudshield_ep3_metadata.json

Los modelos binarios no se publican en GitHub. Deben obtenerse
por un medio autorizado o generarse mediante el procedimiento de
entrenamiento correspondiente.

La API no iniciara correctamente si falta alguno de estos archivos.

## Configuracion de seguridad

Antes de iniciar FastAPI, definir una API Key en la misma terminal:

```powershell
$env:KIPU_API_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
```

No subir claves, tokens ni credenciales al repositorio.

## Ejecutar la API

```powershell
python -m uvicorn modelo.api:app --reload --port 8000
```

Documentacion interactiva:

http://127.0.0.1:8000/docs

## Endpoints

- GET /health: comprueba el estado del servicio.
- POST /predict: realiza predicciones con el modelo local.
- POST /predict/sagemaker: invoca el endpoint de AWS SageMaker.

Los dos endpoints POST requieren el encabezado X-API-Key y
tienen un limite de 5 solicitudes por minuto por direccion IP.

Las transacciones requieren 12 variables, definidas en
la clase Transaccion de modelo/api.py.

## Configuracion de AWS SageMaker

Valores predeterminados:

- Region: us-east-1
- Endpoint: kipu-fraudshield-ep3-endpoint-v3

Se pueden modificar mediante variables de entorno:

```powershell
$env:AWS_DEFAULT_REGION = "us-east-1"
$env:KIPU_SAGEMAKER_ENDPOINT = "kipu-fraudshield-ep3-endpoint-v3"
```

Boto3 obtiene las credenciales mediante sus mecanismos habituales
de autenticacion. En desarrollo local se pueden utilizar
credenciales temporales en el perfil AWS, fuera del repositorio.

El endpoint pertenece a una cuenta AWS concreta.
Para utilizarlo desde otra cuenta se necesitan los permisos y
la configuracion de acceso correspondientes. Cambiar solamente
las variables de entorno no concede acceso entre cuentas.

IMPORTANTE: La implementacion actual carga el modelo local al
iniciar FastAPI, incluso si se utilizara solo /predict/sagemaker.

## Pruebas automatizadas

```powershell
python -m pytest tests/ -q
```

## Seguridad y datos

No subir al repositorio:

- Credenciales o tokens AWS.
- API Keys.
- Datos originales o procesados sensibles.
- Archivos binarios de modelos y paquetes de inferencia.
- Copias de seguridad locales.

Consultar .gitignore antes de preparar cambios para GitHub.

## Trabajo colaborativo

El proyecto se desarrolla en equipo. Antes de integrar cambios
de distintas ramas o entornos AWS, revisar dependencias,
contratos de API y permisos de acceso a los recursos.
