import os
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureOpenAISettings

Kernel = Kernel()

Kernel.add_service(
    AzureChatCompletionService(
)