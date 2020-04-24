from abc import ABC, abstractmethod


class IAgent(ABC):
    @abstractmethod
    def act(self, obs, deterministic=False):
        pass

    @abstractmethod
    def reset(self):
        pass


class IActorCritic(ABC):
    @abstractmethod
    def infer_value(self, feature_tensor):
        pass

    @abstractmethod
    def infer_action_dist(self, feature_tensor):
        pass


class IPolicy(ABC):
    @abstractmethod
    def action_dist(self, obs):
        pass
