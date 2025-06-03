import torch


class SAM(torch.optim.Optimizer):
    """Sharpness Aware Minimization (SAM) optimizer

    Reference: Sharpness-Aware Minimization for Efficiently Improving Generalization
    https://arxiv.org/abs/2010.01412
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: 모델 파라미터
            base_optimizer: 기본 optimizer 클래스 (예: torch.optim.AdamW)
            rho: 근사 이웃의 크기 (perturbation size)
            adaptive: True이면 ASAM (Adaptive SAM) 사용
            **kwargs: base optimizer에 전달할 추가 인자
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        """SAM의 첫 번째 단계: gradient 방향으로 perturbation 적용"""
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # gradient 저장
                self.state[p]["old_p"] = p.data.clone()

                # adaptive scaling
                if group["adaptive"]:
                    # 파라미터별 적응형 스케일링
                    p_norm = p.data.norm(p=2)
                    g_norm = p.grad.data.norm(p=2)
                    scale = group["rho"] * p_norm / (g_norm + 1e-12)

                # perturbation 적용
                e_w = p.grad.data * scale
                p.data.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        """SAM의 두 번째 단계: 원래 위치로 복원 후 실제 업데이트"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 원래 파라미터로 복원
                p.data = self.state[p]["old_p"]

        # base optimizer로 실제 업데이트
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        단일 step 메서드 (closure 필요)
        주의: 일반적으로 first_step()과 second_step()을 명시적으로 호출하는 것을 권장
        """
        assert closure is not None, "SAM requires closure, use first_step and second_step"

        # 첫 번째 forward-backward pass
        closure()
        self.first_step()

        # 두 번째 forward-backward pass
        closure()
        self.second_step()

    def _grad_norm(self):
        """전체 gradient의 norm 계산"""
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        """state dict 로드"""
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def state_dict(self):
        """state dict 저장"""
        state_dict = super().state_dict()
        # base optimizer의 state도 포함
        state_dict['base_optimizer_state'] = self.base_optimizer.state_dict()
        return state_dict


class ESAM(SAM):
    """Efficient SAM (ESAM) - Sharpness 계산 효율화 버전

    매 스텝마다 perturbation을 적용하지 않고 주기적으로만 적용
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False,
                 update_freq=10, **kwargs):
        """
        Args:
            update_freq: SAM 업데이트 빈도 (N 스텝마다 한 번)
        """
        super().__init__(params, base_optimizer, rho, adaptive, **kwargs)
        self.update_freq = update_freq
        self.step_count = 0

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        """주기적으로만 SAM perturbation 적용"""
        self.step_count += 1

        if self.step_count % self.update_freq == 0:
            # 정상적인 SAM 스텝
            super().first_step(zero_grad)
        else:
            # 일반 optimizer처럼 동작
            if zero_grad:
                self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        """주기적으로만 SAM 업데이트 적용"""
        if self.step_count % self.update_freq == 0:
            # 정상적인 SAM 스텝
            super().second_step(zero_grad)
        else:
            # 일반 optimizer 업데이트
            self.base_optimizer.step()
            if zero_grad:
                self.zero_grad()


class LookSAM(SAM):
    """LookSAM: k스텝 앞을 보는 SAM 변형

    Reference: When, Why and How Much? Adaptive Learning Rate Scheduling
    for Sharpness-Aware Minimization
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False,
                 alpha=0.5, k=5, **kwargs):
        """
        Args:
            alpha: interpolation 계수
            k: look-ahead 스텝 수
        """
        super().__init__(params, base_optimizer, rho, adaptive, **kwargs)
        self.alpha = alpha
        self.k = k
        self.step_count = 0

        # look-ahead를 위한 파라미터 복사본
        self.la_params = []
        for group in self.param_groups:
            la_group = []
            for p in group["params"]:
                la_p = p.clone().detach()
                la_group.append(la_p)
            self.la_params.append(la_group)

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        """Look-ahead 메커니즘을 포함한 업데이트"""
        self.step_count += 1

        # 원래 파라미터로 복원
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        # base optimizer로 업데이트
        self.base_optimizer.step()

        # Look-ahead 업데이트
        if self.step_count % self.k == 0:
            for group_idx, group in enumerate(self.param_groups):
                for p_idx, p in enumerate(group["params"]):
                    # interpolation
                    p.data = (self.alpha * p.data +
                              (1 - self.alpha) * self.la_params[group_idx][p_idx])
                    # look-ahead 파라미터 업데이트
                    self.la_params[group_idx][p_idx] = p.data.clone()

        if zero_grad:
            self.zero_grad()