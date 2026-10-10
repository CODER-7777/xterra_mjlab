import torch
from mjlab.managers import RewardTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.string import resolve_matching_names_values

class _TargetCost(ManagerTermBase):
    def __init__(self,cfg,env):
        super().__init__(env)
        self.action_name=cfg.params.get("action_name","joint_pos")
        if list(env.action_manager.active_terms)!= [self.action_name]:
            raise ValueError("Revisit action slicing if more action terms are added ")
        self.term=env.action_manager.get_term(self.action_name)
        if self.term.cfg.clip is None:
            raise ValueError("Configure position target clipping before these costs") #clipping safety check
        template=env.action_manager.action
        self.scale=torch.as_tensor(self.term.scale,device=template.device,dtype=template.dtype)
        if bool((self.scale <=0).any()):
            raise ValueError("Expected positive action scales")
        ids,_,bounds = resolve_matching_names_values(self.term.cfg.clip,self.term.target_names)
        if set(ids) != set(range(self.term.action_dim)):
            raise ValueError("Every position target must have a configured bound")
        limits=torch.tensor(bounds,device=template.device,dtype=template.dtype)
        self.lower=torch.empty_like(template)
        self.upper=torch.empty_like(template)
        self.lower[:,ids]=limits[:,0]
        self.upper[:,ids]=limits[:,1]

    def requested(self,action):
        return action *self.scale+self.term.offset
    def commanded(self,action):
        return torch.clamp(
            self.requested(action),min=self.lower,max=self.upper
        )

class TargetSecondDifference(_TargetCost):
     def __call__(self, env, action_name="joint_pos"):
        if action_name != self.action_name:
            raise ValueError("Action term changed")
        manager=env.action_manager
        q_t=self.commanded(manager.action)
        q_prev= self.commanded(manager.prev_action)
        q_prev_prev=self.commanded(manager.prev_prev_action)
        delta_2 = (q_t - 2*q_prev+q_prev_prev)/self.scale
        cost=delta_2.square().sum(dim=1);
        return torch.where(env.episode_length_buf >=3,cost,torch.zeros_like(cost))

class TargetSaturationPressure(_TargetCost):
    def __call__(self, env, action_name="joint_pos"):
        if action_name != self.action_name:
            raise ValueError("Action term changed")
        requested=self.requested(env.action_manager.action)
        commanded=torch.clamp(requested,min=self.lower,max=self.upper)
        return ((requested-commanded)/self.scale).square().sum(dim=1)


def config_jitter_rewards(cfg,second_weight=-0.01,satuaration_weight=-0.01):
    if second_weight>0 or satuaration_weight>0:
        raise ValueError("Use non positive weights for these costs")
    cfg.rewards["target_second_difference_l2"]=RewardTermCfg(
        func=TargetSecondDifference, weight=second_weight,
        params={"action_name" : "joint_pos"},
    )
    cfg.rewards["target_saturation_l2"]= RewardTermCfg(
        func=TargetSaturationPressure,weight=satuaration_weight,
        params={"action_name" : "joint_pos"}
    )
    return cfg
