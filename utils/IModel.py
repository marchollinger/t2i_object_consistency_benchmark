from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def invoke(prompt):
        pass

