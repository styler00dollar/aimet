# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
import torch
import tempfile
import os
import json
import pytest
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_torch.quantsim import load_encodings_to_sim
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.quantization.encoding_analyzer import PercentileEncodingAnalyzer
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantization.affine import AffineQuantizerBase
from aimet_torch.v2.nn import BaseQuantizationMixin
from ..models_ import test_models

def encodings_are_close(quantizer_1: AffineQuantizerBase, quantizer_2: AffineQuantizerBase):
    min_1, max_1 = quantizer_1.get_min(), quantizer_1.get_max()
    min_2, max_2 = quantizer_2.get_min(), quantizer_2.get_max()
    return torch.allclose(min_1, min_2) \
           and torch.allclose(max_1, max_2) \
           and quantizer_1.bitwidth == quantizer_2.bitwidth \
           and quantizer_1.symmetric == quantizer_2.symmetric


class TestPercentileScheme:
    """ Test Percentile quantization scheme """

    def test_set_percentile_value(self):
        """ Test pecentile scheme by setting different percentile values """

        model = test_models.BasicConv2d(kernel_size=3)
        dummy_input = torch.rand(1, 64, 16, 16)

        def forward_pass(model, args):
            model.eval()
            model(dummy_input)

        sim = QuantizationSimModel(model, dummy_input, quant_scheme="percentile")
        weight_quantizer = sim.model.conv.param_quantizers["weight"]
        assert isinstance(weight_quantizer.encoding_analyzer, PercentileEncodingAnalyzer)

        sim.set_percentile_value(99.9)
        assert weight_quantizer.encoding_analyzer.percentile == 99.9

        sim.compute_encodings(forward_pass, None)
        weight_max_99p9 = weight_quantizer.get_max()

        sim.set_percentile_value(90.0)
        assert weight_quantizer.encoding_analyzer.percentile == 90.0
        sim.compute_encodings(forward_pass, None)
        weight_max_90p0 = weight_quantizer.get_max()

        assert torch.all(weight_max_99p9.gt(weight_max_90p0))

    @pytest.mark.parametrize("config_file", (None, get_path_for_per_channel_config()))
    def test_set_and_freeze_param_encodings(self, config_file):
        model = test_models.BasicConv2d(kernel_size=3)
        dummy_input = torch.rand(1, 64, 16, 16)
        sim = QuantizationSimModel(model, dummy_input, config_file=config_file)
        sim.compute_encodings(lambda model, _: model(dummy_input), None)

        with tempfile.TemporaryDirectory() as temp_dir:
            fname = "test_model"
            sim.export(temp_dir, fname, dummy_input)
            file_path = os.path.join(temp_dir, fname + '.encodings')

            sim_2 = QuantizationSimModel(model, dummy_input, config_file=config_file)

            """
            When: call set_and_freeze_param_encodigns
            Then: Encodings should match
            """
            sim_2.set_and_freeze_param_encodings(file_path)
            assert encodings_are_close(sim.model.conv.param_quantizers["weight"], sim_2.model.conv.param_quantizers["weight"])

        """
        When: Recompute encodings with new weights
        Then: Weight encodings should NOT get overwritten by compute_encodings
        """
        weight_min = sim_2.model.conv.param_quantizers['weight'].min.clone().detach()
        weight_max = sim_2.model.conv.param_quantizers['weight'].max.clone().detach()

        with torch.no_grad():
            sim_2.model.conv.weight.mul_(10)

        sim_2.compute_encodings(lambda model, _: model(dummy_input), None)
        assert torch.equal(weight_min, sim_2.model.conv.param_quantizers['weight'].min)
        assert torch.equal(weight_max, sim_2.model.conv.param_quantizers['weight'].max)

        """
        When: Recompute encodings with new input
        Then: Activation encodings should be updated for the new input (freezing only takes effect to weight quantizers)
        """
        new_dummy_input = 10 * dummy_input
        input_min = sim_2.model.conv.input_quantizers[0].min.clone().detach()
        input_max = sim_2.model.conv.input_quantizers[0].max.clone().detach()
        sim_2.compute_encodings(lambda model, _: model(new_dummy_input), None)
        assert torch.allclose(input_min * 10, sim_2.model.conv.input_quantizers[0].min)
        assert torch.allclose(input_max * 10, sim_2.model.conv.input_quantizers[0].max)

    @pytest.mark.parametrize("config_file", (None, get_path_for_per_channel_config()))
    def test_load_and_freeze_encodings(self, config_file):
        model = test_models.TinyModel()
        dummy_input = torch.rand(1, 3, 32, 32)
        sim = QuantizationSimModel(model, dummy_input, config_file=config_file)
        sim.compute_encodings(lambda model, _: model(dummy_input), None)

        with tempfile.TemporaryDirectory() as temp_dir:
            fname = "test_model"
            sim.export(temp_dir, fname, dummy_input)
            file_path = os.path.join(temp_dir, fname + '_torch.encodings')

            """
            When: Load encodings with ``load_and_freeze_encodings``
            Then: No quantizers should get additionally enabled/disabled
            """
            sim_2 = QuantizationSimModel(test_models.TinyModel(), dummy_input, config_file=config_file)
            all_quantizers = [q for q in sim_2.model.modules() if isinstance(q, QuantizerBase)]
            sim_2.load_and_freeze_encodings(file_path)
            assert all_quantizers == [q for q in sim_2.model.modules() if isinstance(q, QuantizerBase)]

        """
        When: Recompute encodings with new weights
        Then: Weight encodings should NOT get overwritten by compute_encodings
        """
        weight_min = sim_2.model.conv1.param_quantizers['weight'].min.clone().detach()
        weight_max = sim_2.model.conv1.param_quantizers['weight'].max.clone().detach()

        with torch.no_grad():
            sim_2.model.conv1.weight.mul_(10)

        sim_2.compute_encodings(lambda model, _: model(dummy_input), None)
        assert torch.equal(weight_min, sim_2.model.conv1.param_quantizers['weight'].min)
        assert torch.equal(weight_max, sim_2.model.conv1.param_quantizers['weight'].max)

        """
        When: Recompute encodings with new input
        Then: Activation encodings should NOT get overwritten by compute_encodings
        """
        new_dummy_input = 10 * dummy_input
        input_min = sim_2.model.conv1.input_quantizers[0].min.clone().detach()
        input_max = sim_2.model.conv1.input_quantizers[0].max.clone().detach()
        sim_2.compute_encodings(lambda model, _: model(new_dummy_input), None)
        assert torch.equal(input_min, sim_2.model.conv1.input_quantizers[0].min)
        assert torch.equal(input_max, sim_2.model.conv1.input_quantizers[0].max)

    def test_load_and_freeze_with_partial_encodings(self):
        """ Test load_and_freeze encoding API with partial_encodings """
        model = test_models.TinyModel()
        dummy_input = torch.randn(1, 3, 32, 32)

        sample_encoding = {"min": -4, "max": 4, "scale": 0.03, "offset": 8,
                           "bitwidth": 8, "is_symmetric": "False", "dtype": "int"}

        partial_encodings = {
            "activation_encodings": {
                "conv1": {
                    "input": {"0": sample_encoding}
                }
            },
            "param_encodings": {"conv1.weight": [sample_encoding]}
        }

        sim = QuantizationSimModel(model, dummy_input)
        all_quantizers = [q for q in sim.model.modules() if isinstance(q, QuantizerBase)]
        sim.load_and_freeze_encodings(partial_encodings)

        """
        When: Load partial encodings with ``load_and_freeze_encodings``
        Then: No quantizers should get additionally enabled/disabled
        """
        assert all_quantizers == [q for q in sim.model.modules() if isinstance(q, QuantizerBase)]

        """
        When: Recompute encodings with new weights
        Then: Weight encodings imported from the config file should NOT get overwritten by compute_encodings
            2) Weight encodings NOT imported from the config file SHOULD get overwritten by compute_encodings
        """
        conv1_weight_min = sim.model.conv1.param_quantizers['weight'].min.clone().detach()
        conv1_weight_max = sim.model.conv1.param_quantizers['weight'].max.clone().detach()
        with torch.no_grad():
            sim.model.conv1.weight.mul_(10)

        sim.compute_encodings(lambda model, _: model(dummy_input), None)
        assert torch.equal(conv1_weight_min, sim.model.conv1.param_quantizers['weight'].min)
        assert torch.equal(conv1_weight_max, sim.model.conv1.param_quantizers['weight'].max)

        """
        When: Recompute encodings with new weights
        Then: Weight encodings NOT imported from the config file SHOULD get overwritten by compute_encodings
        """
        fc_weight_min = sim.model.fc.param_quantizers['weight'].min.clone().detach()
        fc_weight_max = sim.model.fc.param_quantizers['weight'].max.clone().detach()
        with torch.no_grad():
            sim.model.fc.weight.mul_(10)
        sim.compute_encodings(lambda model, _: model(dummy_input), None)
        assert torch.allclose(fc_weight_min * 10, sim.model.fc.param_quantizers['weight'].min)
        assert torch.allclose(fc_weight_max * 10, sim.model.fc.param_quantizers['weight'].max)

        """
        When: Recompute encodings with new input
        Then: Activation encodings should NOT get overwritten by compute_encodings
            1) Activation encodings imported from the config file should NOT get overwritten by compute_encodings
            2) Activation encodings NOT imported from the config file SHOULD get overwritten by compute_encodings
        """
        new_dummy_input = 10 * dummy_input
        conv1_input_min = sim.model.conv1.input_quantizers[0].min.clone().detach()
        conv1_input_max = sim.model.conv1.input_quantizers[0].max.clone().detach()
        fc_output_min = sim.model.fc.output_quantizers[0].min.clone().detach()
        fc_output_max = sim.model.fc.output_quantizers[0].max.clone().detach()
        sim.compute_encodings(lambda model, _: model(new_dummy_input), None)
        assert torch.equal(conv1_input_min, sim.model.conv1.input_quantizers[0].min)
        assert torch.equal(conv1_input_max, sim.model.conv1.input_quantizers[0].max)
        assert not any(torch.isclose(fc_output_min, sim.model.fc.output_quantizers[0].min))
        assert not any(torch.isclose(fc_output_max, sim.model.fc.output_quantizers[0].max))

    def test_load_encodings(self):
        model = test_models.TinyModel()
        dummy_input = torch.randn(1, 3, 32, 32)

        sample_encoding = {"min": -4, "max": 4, "scale": 0.03, "offset": 8,
                           "bitwidth": 8, "is_symmetric": "False", "dtype": "int"}
        sample_encoding2 = {"min": -8, "max": 8, "scale": 0.06, "offset": 8,
                            "bitwidth": 8, "is_symmetric": "False", "dtype": "int"}

        encodings = {
            "activation_encodings": {
                "conv1": {
                    "input": {"0": sample_encoding}
                }
            },
            "param_encodings": {"conv1.weight": [sample_encoding]}
        }
        encodings2 = {
            "activation_encodings": {
                "conv1": {
                    "input": {"0": sample_encoding2}
                }
            },
            "param_encodings": {"conv1.weight": [sample_encoding2]}
        }

        """
        When: Call load_encodings with partial=False
        Then: All the dangling quantizers should be removed
        """
        sim = QuantizationSimModel(model, dummy_input)
        sim.load_encodings(encodings, partial=False)
        all_quantizers = [q for q in sim.model.modules() if isinstance(q, QuantizerBase)]
        assert all_quantizers == [sim.model.conv1.param_quantizers['weight'],
                                  sim.model.conv1.input_quantizers[0]]

        """
        When: Call load_encodings with partial=True
        Then: No quantizer gets removed
        """
        sim = QuantizationSimModel(model, dummy_input)
        all_quantizers = [q for q in sim.model.modules() if isinstance(q, QuantizerBase)]
        sim.load_encodings(encodings, partial=True)
        assert all_quantizers == [q for q in sim.model.modules() if isinstance(q, QuantizerBase)]

        for requires_grad in (True, False):
            """
            When: Call load_encodings with requires_grad specified
            Then: The loaded quantizers should be set to requires_grad=True/False accordingly
            """
            sim = QuantizationSimModel(model, dummy_input)
            all_parameters = {
                q: (q.min.clone(), q.max.clone())
                for q in sim.model.modules() if isinstance(q, QuantizerBase)
            }
            sim.load_encodings(encodings, requires_grad=requires_grad)
            assert sim.model.conv1.param_quantizers['weight'].min.requires_grad ==\
                   sim.model.conv1.param_quantizers['weight'].max.requires_grad ==\
                   requires_grad
            assert sim.model.conv1.input_quantizers[0].min.requires_grad ==\
                   sim.model.conv1.input_quantizers[0].max.requires_grad ==\
                   requires_grad

            # requires_grad of all the oither quantization parameters should not be modified
            for q, (min_copy, max_copy) in all_parameters.items():
                if q in (sim.model.conv1.param_quantizers['weight'],
                         sim.model.conv1.input_quantizers[0]):
                    continue
                assert q.min.requires_grad == min_copy.requires_grad
                assert q.max.requires_grad == max_copy.requires_grad

            """
            When: Call load_encodings with requires_grad NOT specified
            Then: requires_grad flag should be kept unchanged
            """
            sim.load_encodings(encodings, requires_grad=None)
            assert sim.model.conv1.param_quantizers['weight'].min.requires_grad ==\
                   sim.model.conv1.param_quantizers['weight'].max.requires_grad ==\
                   requires_grad
            assert sim.model.conv1.input_quantizers[0].min.requires_grad ==\
                   sim.model.conv1.input_quantizers[0].max.requires_grad ==\
                   requires_grad

            # requires_grad of all the oither quantization parameters should not be modified
            for q, (min_copy, max_copy) in all_parameters.items():
                if q in (sim.model.conv1.param_quantizers['weight'],
                         sim.model.conv1.input_quantizers[0]):
                    continue
                assert q.min.requires_grad == min_copy.requires_grad
                assert q.max.requires_grad == max_copy.requires_grad

        """
        When: Call load_encodings with allow_overwrite=True
        Then: The loaded quantizers should be overwritten by a subsequent
              compute_encodings or load_encodings
        """
        sim = QuantizationSimModel(model, dummy_input)
        sim.load_encodings(encodings, allow_overwrite=True)
        weight_min = sim.model.conv1.param_quantizers['weight'].min.clone().detach()
        weight_max = sim.model.conv1.param_quantizers['weight'].max.clone().detach()
        input_min = sim.model.conv1.input_quantizers[0].min.clone().detach()
        input_max = sim.model.conv1.input_quantizers[0].max.clone().detach()

        sim.compute_encodings(lambda model, _: model(dummy_input), None)

        assert not any(torch.isclose(weight_min,
                                     sim.model.conv1.param_quantizers['weight'].min))
        assert not any(torch.isclose(weight_max,
                                     sim.model.conv1.param_quantizers['weight'].max))
        assert not any(torch.isclose(input_min,
                                     sim.model.conv1.input_quantizers[0].min))
        assert not any(torch.isclose(input_max,
                                     sim.model.conv1.input_quantizers[0].max))

        weight_min = sim.model.conv1.param_quantizers['weight'].min.clone().detach()
        weight_max = sim.model.conv1.param_quantizers['weight'].max.clone().detach()
        input_min = sim.model.conv1.input_quantizers[0].min.clone().detach()
        input_max = sim.model.conv1.input_quantizers[0].max.clone().detach()

        sim.load_encodings(encodings2)

        assert not any(torch.isclose(weight_min,
                                     sim.model.conv1.param_quantizers['weight'].min))
        assert not any(torch.isclose(weight_max,
                                     sim.model.conv1.param_quantizers['weight'].max))
        assert not any(torch.isclose(input_min,
                                     sim.model.conv1.input_quantizers[0].min))
        assert not any(torch.isclose(input_max,
                                     sim.model.conv1.input_quantizers[0].max))

        """
        When: Call load_encodings with allow_overwrite=False
        Then: The loaded quantizers should NOT be overwritten by a subsequent
              compute_encodings or load_encodings
        """
        sim = QuantizationSimModel(model, dummy_input)
        sim.load_encodings(encodings, allow_overwrite=False)
        weight_min = sim.model.conv1.param_quantizers['weight'].min.clone().detach()
        weight_max = sim.model.conv1.param_quantizers['weight'].max.clone().detach()
        input_min = sim.model.conv1.input_quantizers[0].min.clone().detach()
        input_max = sim.model.conv1.input_quantizers[0].max.clone().detach()

        sim.compute_encodings(lambda model, _: model(dummy_input), None)

        assert torch.equal(weight_min, sim.model.conv1.param_quantizers['weight'].min)
        assert torch.equal(weight_max, sim.model.conv1.param_quantizers['weight'].max)
        assert torch.equal(input_min, sim.model.conv1.input_quantizers[0].min)
        assert torch.equal(input_max, sim.model.conv1.input_quantizers[0].max)

        sim.load_encodings(encodings2)

        assert torch.equal(weight_min, sim.model.conv1.param_quantizers['weight'].min)
        assert torch.equal(weight_max, sim.model.conv1.param_quantizers['weight'].max)
        assert torch.equal(input_min, sim.model.conv1.input_quantizers[0].min)
        assert torch.equal(input_max, sim.model.conv1.input_quantizers[0].max)

        """
        When: Call load_encodings with allow_overwrite=None
        Then: Whether the loaded quantizers can be overwritten is kept unchanged
        """
        sim.load_encodings(encodings, allow_overwrite=None)

        assert torch.equal(weight_min, sim.model.conv1.param_quantizers['weight'].min)
        assert torch.equal(weight_max, sim.model.conv1.param_quantizers['weight'].max)
        assert torch.equal(input_min, sim.model.conv1.input_quantizers[0].min)
        assert torch.equal(input_max, sim.model.conv1.input_quantizers[0].max)

    @pytest.mark.parametrize('load_encodings_fn', [load_encodings_to_sim,
                                                   QuantizationSimModel.load_and_freeze_encodings,
                                                   QuantizationSimModel.set_and_freeze_param_encodings])
    def test_legacy_load_encodings_partial_encoding(self, load_encodings_fn):
        model = test_models.SmallMnist()
        dummy_input = torch.rand(1, 1, 28, 28)

        partial_torch_encodings = {
            "activation_encodings": {
                "conv1": {
                    "input": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.9978924989700317,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.003913303837180138
                        }
                    }
                },
                "conv2": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.4923851788043976,
                            "min": -0.43767568469047546,
                            "offset": -120,
                            "scale": 0.0036472973879426718
                        }
                    }
                },
                "fc2": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.1948324590921402,
                            "min": -0.15752412378787994,
                            "offset": -114,
                            "scale": 0.0013817904982715845
                        }
                    }
                },
                "relu1": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 1.0608084201812744,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.004160033073276281
                        }
                    }
                },
                "relu3": {
                    "output": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.5247029066085815,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.0020576585084199905
                        }
                    }
                }
            },
            "excluded_layers": [],
            "param_encodings": {
                "conv1.weight": [
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.18757757544517517,
                        "min": -0.2143743634223938,
                        "offset": -8,
                        "scale": 0.026796795427799225
                    }
                ],
                "fc2.weight": [
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.13095608353614807,
                        "min": -0.14966410398483276,
                        "offset": -8,
                        "scale": 0.018708012998104095
                    }
                ]
            },
            "quantizer_args": {
                "activation_bitwidth": 8,
                "dtype": "int",
                "is_symmetric": True,
                "param_bitwidth": 4,
                "per_channel_quantization": False,
                "quant_scheme": "post_training_tf_enhanced"
            },
            "version": "0.6.1"
        }

        qsim = QuantizationSimModel(model, dummy_input)
        quantizers = [q for q in qsim.model.modules() if isinstance(q, QuantizerBase)]

        with tempfile.TemporaryDirectory() as temp_dir:
            fname = os.path.join(temp_dir, "temp_partial_torch_encodings.encodings")
            with open(fname, 'w') as f:
                json.dump(partial_torch_encodings, f)

            load_encodings_fn(qsim, fname)

        if load_encodings_fn is load_encodings_to_sim:
            """
            When: Load partial encodings with load_encodings_to_sim
            Then: Quantizers that have no corresponding encodings should be removed
            """
            loaded_quantizers = [
                qsim.model.conv1.input_quantizers[0],
                qsim.model.conv1.param_quantizers['weight'],
                qsim.model.conv2.output_quantizers[0],
                qsim.model.fc2.output_quantizers[0],
                qsim.model.fc2.param_quantizers['weight'],
                qsim.model.relu1.output_quantizers[0],
                qsim.model.relu3.output_quantizers[0],
            ]
            assert sorted(loaded_quantizers, key=id) ==\
                   sorted([q for q in qsim.model.modules() if isinstance(q, QuantizerBase)], key=id)

        elif load_encodings_fn in [QuantizationSimModel.load_and_freeze_encodings,
                                   QuantizationSimModel.set_and_freeze_param_encodings]:
            """
            When: Load partial encodings with load_and_freeze_encodings or set_and_freeze_param_encodings
            Then: Quantizers shouldn't be additionally removed or instantiated
            """
            assert quantizers == [q for q in qsim.model.modules() if isinstance(q, QuantizerBase)]
        else:
            raise AssertionError

    @pytest.mark.parametrize('load_encodings_fn', [load_encodings_to_sim,
                                                   QuantizationSimModel.load_and_freeze_encodings,
                                                   QuantizationSimModel.set_and_freeze_param_encodings])
    def test_legacy_load_encodings_mismatching_encoding(self, load_encodings_fn):
        model = test_models.SmallMnist()
        dummy_input = torch.rand(1, 1, 28, 28)

        invalid_torch_encodings = {
            "excluded_layers": [],
            "activation_encodings": {
                "conv999": {
                    "input": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.9978924989700317,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.003913303837180138
                        }
                    }
                },
            },
            "param_encodings": {
                "conv999.weight": [ # NOTE: conv999 does not exist in the model
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.18757757544517517,
                        "min": -0.2143743634223938,
                        "offset": -8,
                        "scale": 0.026796795427799225
                    }
                ],
            },
            "quantizer_args": {
                "activation_bitwidth": 8,
                "dtype": "int",
                "is_symmetric": True,
                "param_bitwidth": 4,
                "per_channel_quantization": False,
                "quant_scheme": "post_training_tf_enhanced"
            },
            "version": "0.6.1"
        }

        qsim = QuantizationSimModel(model, dummy_input)

        """
        When: Try to load encoding file some keys of which are missing in the model
              (Note that conv999 does not exist in the model)
        Then: Throw runtime error
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            fname = os.path.join(temp_dir, "temp_partial_torch_encodings.encodings")
            with open(fname, 'w') as f:
                json.dump(invalid_torch_encodings, f)

            with pytest.raises(RuntimeError):
                load_encodings_fn(qsim, fname)

    @pytest.mark.parametrize('load_encodings_fn', [load_encodings_to_sim,
                                                   QuantizationSimModel.load_and_freeze_encodings,
                                                   QuantizationSimModel.set_and_freeze_param_encodings])
    def test_legacy_load_encodings_to_disabled_quantizer(self, load_encodings_fn):
        model = test_models.SmallMnist()
        dummy_input = torch.rand(1, 1, 28, 28)

        invalid_torch_encodings = {
            "excluded_layers": [],
            "activation_encodings": {
                "conv1": {
                    "input": {
                        "0": {
                            "bitwidth": 8,
                            "dtype": "int",
                            "is_symmetric": "False",
                            "max": 0.9978924989700317,
                            "min": 0.0,
                            "offset": 0,
                            "scale": 0.003913303837180138
                        }
                    }
                },
            },
            "param_encodings": {
                "conv1.weight": [
                    {
                        "bitwidth": 4,
                        "dtype": "int",
                        "is_symmetric": "True",
                        "max": 0.18757757544517517,
                        "min": -0.2143743634223938,
                        "offset": -8,
                        "scale": 0.026796795427799225
                    }
                ],
            },
            "quantizer_args": {
                "activation_bitwidth": 8,
                "dtype": "int",
                "is_symmetric": True,
                "param_bitwidth": 4,
                "per_channel_quantization": False,
                "quant_scheme": "post_training_tf_enhanced"
            },
            "version": "0.6.1"
        }

        qsim = QuantizationSimModel(model, dummy_input)

        """
        Given: Input/param quantizers of conv1 is disabled
        When: Try to load input/param quantizers to conv1
        Then: Throw runtime error
        """
        qsim.model.conv1.input_quantizers[0] = None
        qsim.model.conv1.param_quantizers['weight'] = None

        with tempfile.TemporaryDirectory() as temp_dir:
            fname = os.path.join(temp_dir, "temp_partial_torch_encodings.encodings")
            with open(fname, 'w') as f:
                json.dump(invalid_torch_encodings, f)

            with pytest.raises(RuntimeError):
                load_encodings_fn(qsim, fname)

class TestQuantsimUtilities:

    def test_populate_marker_map(self):
        model = test_models.BasicConv2d(kernel_size=3)
        dummy_input = torch.rand(1, 64, 16, 16)
        sim = QuantizationSimModel(model, dummy_input)
        conv_layer = sim.model.conv
        for name, module in sim.model.named_modules():
            if module is conv_layer:
                conv_name = name
                break
        assert conv_name not in sim._module_marker_map.keys()
        sim.run_modules_for_traced_custom_marker([conv_layer], dummy_input)
        assert conv_name in sim._module_marker_map.keys()
        assert torch.equal(sim._module_marker_map[conv_name](dummy_input), conv_layer.get_original_module()(dummy_input))

    def test_get_qc_quantized_modules(self):
        model = test_models.BasicConv2d(kernel_size=3)
        dummy_input = torch.rand(1, 64, 16, 16)
        sim = QuantizationSimModel(model, dummy_input)
        conv_layer = sim.model.conv
        assert ("conv", conv_layer) in sim._get_qc_quantized_layers(sim.model)

    def test_get_leaf_module_to_name_map(self):
        model = test_models.NestedConditional()
        dummy_input = torch.rand(1, 3), torch.tensor([True])
        sim = QuantizationSimModel(model, dummy_input)
        leaf_modules = sim._get_leaf_module_to_name_map()
        for name, module in sim.model.named_modules():
            if isinstance(module, BaseQuantizationMixin):
                assert module in leaf_modules.keys()
                assert leaf_modules[module] == name
