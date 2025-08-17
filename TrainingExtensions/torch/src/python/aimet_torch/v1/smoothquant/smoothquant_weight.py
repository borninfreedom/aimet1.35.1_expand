# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023-2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

"""Implementation of SmoothQuant for PyTorch models"""

import os
import contextlib
import itertools
import json
import tempfile
from typing import Tuple, Union, Dict, List, Callable, Any, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_torch import utils
from aimet_torch.v1.quantsim import QuantizationSimModel, StaticGridQuantWrapper
from aimet_torch.meta import connectedgraph_utils
from aimet_torch.utils import get_ordered_list_of_modules, CachedDataset

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

SmoothQuantSupportedModules = (torch.nn.Conv2d, torch.nn.Linear)

class SmoothQuantParameters:
    """Configuration parameters for SmoothQuant"""
    
    def __init__(self, 
                 data_loader: DataLoader,
                 num_batches: int,
                 alpha: float = 0.5,
                 percentile: float = 99.9,
                 channel_wise: bool = True,
                 use_empirical_scaling: bool = True,
                 forward_fn: Callable[[torch.nn.Module, Any], Any] = None):
        """
        :param data_loader: Data loader for calibration
        :param num_batches: Number of batches for calibration
        :param alpha: Trade-off parameter between activation and weight scaling (default: 0.5)
        :param percentile: Percentile to use for activation scaling (default: 99.9)
        :param channel_wise: Whether to compute scales per channel (default: True)
        :param use_empirical_scaling: Whether to use empirical scaling for activations (default: True)
        :param forward_fn: Optional adapter function for custom forward pass
        """
        if len(data_loader) < num_batches:
            raise ValueError(f'Cannot fetch {num_batches} batches from '
                           f'data loader of length {len(data_loader)}')
        
        self.data_loader = data_loader
        self.num_batches = num_batches
        self.alpha = alpha
        self.percentile = percentile
        self.channel_wise = channel_wise
        self.use_empirical_scaling = use_empirical_scaling
        self.forward_fn = forward_fn

class SmoothQuantWrapper(StaticGridQuantWrapper):
    """Wrapper for modules that need to be smooth quantized"""
    
    def __init__(self, module_to_wrap: torch.nn.Module, scale_acts: torch.Tensor):
        """
        :param module_to_wrap: The module to be wrapped
        :param scale_acts: Scaling factors for activations
        """
        super().__init__(module_to_wrap)
        
        # Apply scaling directly to the weight and bias
        with torch.no_grad():
            # Scale the weight
            module_to_wrap.weight.data /= scale_acts.view(1, -1, 1, 1) \
                if isinstance(module_to_wrap, torch.nn.Conv2d) else scale_acts.view(1, -1)
            
            # Scale the bias if it exists
            if module_to_wrap.bias is not None:
                module_to_wrap.bias.data /= scale_acts.view(-1) \
                    if isinstance(module_to_wrap, torch.nn.Conv2d) else scale_acts.view(-1)
        
    def forward(self, *inputs, **kwargs):
        """Forward pass without explicit activation scaling"""
        # No need to scale activations as we've already scaled the weights
        return self._module_to_wrap(*inputs, **kwargs)

class SmoothQuant:
    """Implementation of SmoothQuant for PyTorch"""
    
    @classmethod
    def apply_smooth_quant(cls, model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple],
                          params: SmoothQuantParameters, path: str, filename_prefix: str,
                          default_param_bw: int = 8,
                          default_quant_scheme: QuantScheme = QuantScheme.post_training_tf_enhanced,
                          ignored_modules: List[torch.nn.Module] = None) -> torch.nn.Module:
        """
        Apply SmoothQuant to a model and save calibration parameters
        
        :param model: Model to apply SmoothQuant to
        :param dummy_input: Dummy input for model analysis
        :param params: SmoothQuant parameters
        :param path: Path to save calibration parameters
        :param filename_prefix: Prefix for saved files
        :param default_param_bw: Default parameter bitwidth
        :param default_quant_scheme: Quantization scheme to use
        :param ignored_modules: List of modules to ignore during smoothing
        :return: Smoothed and quantized model
        """
        # Create QuantSim model
        quant_sim = cls._create_quantsim(model, dummy_input, default_quant_scheme, default_param_bw)
        
        # Get module ordering using connected graph
        module_act_func_pair = connectedgraph_utils.get_module_act_func_pair(model, dummy_input)
        
        # Apply smoothing
        cls._smooth_quant_model(model, quant_sim, module_act_func_pair, params, ignored_modules)
        
        # Export encodings
        cls._export_encodings(path, filename_prefix, quant_sim)
        
        return quant_sim.model

    @classmethod
    def _smooth_quant_model(cls, model: torch.nn.Module, quant_sim: QuantizationSimModel,
                           module_act_func_pair: Dict, params: SmoothQuantParameters,
                           ignored_modules: List[torch.nn.Module] = None):
        """
        Apply SmoothQuant to all supported modules in the model.
        The activation scales are absorbed into the weights to maintain QNN compatibility.
        
        :param model: Original model
        :param quant_sim: Quantization simulation model
        :param module_act_func_pair: Dictionary mapping modules to activation functions
        :param params: SmoothQuant parameters
        :param ignored_modules: Modules to ignore during smoothing
        """
        if ignored_modules is None:
            ignored_modules = []
            
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Cache calibration data
            cached_dataset = CachedDataset(params.data_loader, params.num_batches, tmp_dir)
            
            # Dictionary to store activation statistics
            activation_dict = {}
            
            def hook_fn(module, input_feat, _):
                if module not in activation_dict:
                    activation_dict[module] = []
                activation_dict[module].append(input_feat[0].detach())
            
            # Register hooks for collecting activations
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, SmoothQuantSupportedModules) and module not in ignored_modules:
                    hooks.append(module.register_forward_hook(hook_fn))
            
            # Collect activation statistics
            model.eval()
            with torch.no_grad():
                for batch in tqdm(cached_dataset, desc="Collecting statistics"):
                    if params.forward_fn:
                        params.forward_fn(model, batch)
                    else:
                        model(batch[0])
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # 创建模块到scale的映射
            module_scales = {}
            
            # 第一遍：收集所有模块的scale信息
            for name, module in model.named_modules():
                if isinstance(module, SmoothQuantSupportedModules) and module not in ignored_modules:
                    if module in activation_dict:
                        # 收集该模块输入的统计信息
                        act_stats = torch.cat(activation_dict[module], dim=0)
                        
                        # 计算scale factors
                        scale_weight, scale_activation = cls._compute_scales(
                            module, act_stats, params
                        )
                        
                        # 存储scale信息
                        module_scales[name] = (scale_weight, scale_activation)
            
            # 第二遍：应用scale到前一层的权重上
            prev_module = None
            prev_name = None
            
            for name, module in model.named_modules():
                if isinstance(module, SmoothQuantSupportedModules) and module not in ignored_modules:
                    # 如果当前模块有scale信息
                    if name in module_scales:
                        scale_weight, scale_activation = module_scales[name]
                        
                        # 如果有前一层，将当前层的activation scale应用到前一层的权重上
                        if prev_module is not None and prev_name in module_scales:
                            with torch.no_grad():
                                prev_module.weight.data *= scale_activation
                                if prev_module.bias is not None:
                                    prev_module.bias.data *= scale_activation.view(-1)
                        
                        # 更新前一层信息
                        prev_module = module
                        prev_name = name
                        
                        # Instead of wrapping with activation scaling, we store the optimized parameters
                        # This ensures the ONNX model will have the correct weights without additional ops
                        quant_wrapper = utils.get_named_module(quant_sim.model, name)
                        if isinstance(quant_wrapper, StaticGridQuantWrapper):
                            # Update quantizer settings if needed
                            quant_wrapper.input_quantizers[0].update_encoding_stats(act_stats)
                            quant_wrapper.param_quantizers['weight'].update_encoding_stats(module.weight.data)

    @staticmethod
    def _compute_scales(module: torch.nn.Module, activations: torch.Tensor,
                       params: SmoothQuantParameters) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaling factors for weights and activations.
        The activation scale will be absorbed into the weights to maintain QNN compatibility.
        
        :param module: The module to compute scales for
        :param activations: Collected activation tensor
        :param params: SmoothQuant parameters
        :return: Weight and activation scaling factors
        """
        if params.channel_wise:
            # Channel-wise scaling for activations
            if isinstance(module, torch.nn.Conv2d):
                # For Conv2d, reshape activations to [C, N*H*W]
                act_reshaped = activations.transpose(0, 1).reshape(activations.size(1), -1)
            else:  # Linear
                # For Linear, reshape activations to [C, N]
                act_reshaped = activations.t()
            
            if params.use_empirical_scaling:
                # Use percentile for more robust scaling
                act_scales = torch.tensor([
                    torch.quantile(row.abs(), params.percentile/100)
                    for row in act_reshaped
                ]).to(activations.device)
            else:
                # Use max for scaling
                act_scales = act_reshaped.abs().max(dim=1)[0]
        else:
            # Layer-wise scaling
            if params.use_empirical_scaling:
                act_scales = torch.quantile(activations.abs().flatten(), 
                                         params.percentile/100)
            else:
                act_scales = activations.abs().max()
            # Broadcast to all channels
            act_scales = torch.full((activations.size(1),), 
                                  act_scales.item(), 
                                  device=activations.device)
        
        # Compute weight statistics (always channel-wise)
        weight = module.weight.data
        weight_scales = weight.view(weight.size(0), -1).abs().max(dim=1)[0]
        
        # Compute scaling factors using alpha parameter
        # Note: We absorb the activation scale into the weight scale
        weight_scale_factor = (weight_scales ** params.alpha) * (act_scales ** (1 - params.alpha))
        
        # The activation scale is the reciprocal of the weight scale to maintain the same output
        act_scale_factor = 1.0 / weight_scale_factor
        
        # Reshape scales according to module type
        if isinstance(module, torch.nn.Conv2d):
            weight_scale_factor = weight_scale_factor.view(-1, 1, 1, 1)
            act_scale_factor = act_scale_factor.view(1, -1, 1, 1)
        else:  # Linear
            weight_scale_factor = weight_scale_factor.view(-1, 1)
            act_scale_factor = act_scale_factor.view(1, -1)
            
        return weight_scale_factor, act_scale_factor

    @staticmethod
    def _create_quantsim(model: torch.nn.Module, dummy_input: torch.Tensor,
                        quant_scheme: QuantScheme, param_bw: int) -> QuantizationSimModel:
        """Create a QuantizationSimModel"""
        return QuantizationSimModel(model, dummy_input=dummy_input,
                                  quant_scheme=quant_scheme,
                                  default_param_bw=param_bw)

    @classmethod
    def _wrap_module_with_smooth_quant(cls, model: torch.nn.Module, module_name: str,
                                      scale_activation: torch.Tensor):
        """
        Replace module with SmoothQuantWrapper
        
        :param model: Model containing the module
        :param module_name: Name of module to wrap
        :param scale_activation: Activation scaling factors
        """
        module = utils.get_named_module(model, module_name)
        if isinstance(module, StaticGridQuantWrapper):
            wrapped_module = SmoothQuantWrapper(module._module_to_wrap, scale_activation)
            
            # Get parent module and name
            parent_name, _, target_name = module_name.rpartition('.')
            parent_module = model if not parent_name else utils.get_named_module(model, parent_name)
            
            # Replace module
            setattr(parent_module, target_name, wrapped_module)

    @staticmethod
    def _export_encodings(path: str, filename_prefix: str, quant_sim: QuantizationSimModel):
        """
        Export quantization encodings in standard AIMET format without additional scale parameters
        """
        os.makedirs(path, exist_ok=True)
        encoding_path = os.path.join(path, f"{filename_prefix}.encodings")
        
        # Create standard AIMET encodings dictionary
        encodings = {
            "param_encodings": {},
            "activation_encodings": {}
        }
        
        # Export standard quantization parameters
        for name, module in quant_sim.model.named_modules():
            if isinstance(module, StaticGridQuantWrapper):
                # Export parameter encodings
                for param_name, quantizer in module.param_quantizers.items():
                    if quantizer.encoding is not None:
                        full_param_name = f"{name}.{param_name}"
                        encodings["param_encodings"][full_param_name] = {
                            "min": float(quantizer.encoding.min),
                            "max": float(quantizer.encoding.max),
                            "scale": float(quantizer.encoding.scale),
                            "offset": int(quantizer.encoding.offset),
                            "bitwidth": int(quantizer.bitwidth),
                            "is_symmetric": "True" if quantizer.use_symmetric_encodings else "False"
                        }
                
                # Export activation encodings
                for idx, quantizer in enumerate(module.input_quantizers):
                    if quantizer.encoding is not None:
                        encodings["activation_encodings"][f"{name}.input_{idx}"] = {
                            "min": float(quantizer.encoding.min),
                            "max": float(quantizer.encoding.max),
                            "scale": float(quantizer.encoding.scale),
                            "offset": int(quantizer.encoding.offset),
                            "bitwidth": int(quantizer.bitwidth),
                            "is_symmetric": "True" if quantizer.use_symmetric_encodings else "False"
                        }
        
        # Save encodings
        with open(encoding_path, 'w') as f:
            json.dump(encodings, f, indent=4)
