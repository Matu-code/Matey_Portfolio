from azure.identity import ClientSecretCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerinstance.models import (
    ContainerGroup, Container, ResourceRequests, ResourceRequirements, 
    ContainerGroupNetworkProtocol, OperatingSystemTypes,
    IpAddress, Port
)

# Replace with your own values
SUBSCRIPTION_ID = ''
RESOURCE_GROUP = ''
CONTAINER_NAME = ""
IMAGE = ''  # Docker Hub image
CPU_CORE_COUNT = 1.0
MEMORY_GB = 1.5
TENANT_ID = ""
CLIENT_ID = ""
CLIENT_SECRET = ""

# Get credentials
credentials = ClientSecretCredential(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    tenant_id=TENANT_ID
)

# Create a Container Instance Management client
container_client = ContainerInstanceManagementClient(credentials, SUBSCRIPTION_ID)

# Define the container
container_resource_requests = ResourceRequests(memory_in_gb=MEMORY_GB, cpu=CPU_CORE_COUNT)
container_resource_requirements = ResourceRequirements(requests=container_resource_requests)
container = Container(name=CONTAINER_NAME, image=IMAGE, resources=container_resource_requirements, ports=[Port(port=80)])

# Define the group of containers
container_group = ContainerGroup(
    location='westeurope',
    containers=[container],
    os_type=OperatingSystemTypes.linux,
    ip_address=IpAddress(ports=[Port(protocol=ContainerGroupNetworkProtocol.tcp, port=80)], type='Public')
)

# Create the container group
container_client.container_groups.begin_create_or_update(RESOURCE_GROUP, CONTAINER_NAME, container_group)

print(f"Deployment of {CONTAINER_NAME} started.")
