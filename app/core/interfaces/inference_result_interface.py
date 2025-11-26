from abc import ABC, abstractmethod

class IInferenceResult(ABC):
    @abstractmethod
    def save(self, result_data: dict, filename: str) -> str:
        ...
