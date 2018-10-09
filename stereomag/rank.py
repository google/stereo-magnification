#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantitative evaluation of view synthesis results.

Read in dumped json data and compute various statistics.
"""
import os
import json
import numpy as np
from scipy.stats import wilcoxon
from scipy.stats.mstats import rankdata

from pyglib import app
from pyglib import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('root', 'evaluation', 'Evaluation directory')
flags.DEFINE_string(
    'model_names',
    'v4_1024,v4_1024_alpha,v4_1024_singleRGB,v4_1024_fgbg,v4_1024_all',
    'model names')
flags.DEFINE_string('data_split', 'test', 'split of the data')
flags.DEFINE_string('stats', 'mean,rank,diff,wilcoxon',
                    'which stats to compute')


def load_data(root, model):
  with open(root + '/json/' + model + '.json') as f:
    data = json.load(f)
  return data


def merge_into(data, d):
  if data == {}:
    data['models'] = []
    data['examples'] = d['examples']
    n = len(data['examples'])
    data['ssim'] = [[]] * n
    data['psnr'] = [[]] * n

  for m in d['model_names']:
    assert m not in data['models']
    data['models'].append(str(m))

  assert d['examples'] == data['examples']
  assert len(data['ssim']) == len(d['ssim'])
  assert len(data['psnr']) == len(d['psnr'])
  data['ssim'] = [a + b for (a, b) in zip(data['ssim'], d['ssim'])]
  data['psnr'] = [a + b for (a, b) in zip(data['psnr'], d['psnr'])]


def compute_mean(data):
  print '\nMEAN + STD\n'
  ssim = np.array(data['ssim'])
  psnr = np.array(data['psnr'])
  for i, m in enumerate(data['models']):
    print '%30s    ssim %.3f ± %.3f    psnr %.2f ± %.2f' % (
        m, np.mean(ssim[:, i]), np.std(ssim[:, i]), np.mean(psnr[:, i]),
        np.std(psnr[:, i]))


def compute_rank(data):
  print '\nRANK\n'
  # rankdata assigns rank 1 to the lowest element, so
  # we need to negate before ranking.
  ssim_rank = rankdata(np.array(data['ssim']) * -1.0, axis=1)
  psnr_rank = rankdata(np.array(data['psnr']) * -1.0, axis=1)
  # Rank mean + std.
  for i, m in enumerate(data['models']):
    print '%30s    ssim-rank %.2f ± %.2f    psnr-rank %.2f ± %.2f' % (
        m, np.mean(ssim_rank[:, i]), np.std(ssim_rank[:, i]),
        np.mean(psnr_rank[:, i]), np.std(psnr_rank[:, i]))
  # Rank frequencies
  print '\n    SSIM rank freqs'
  print_rank_freqs(data, ssim_rank)
  print '\n    PSNR rank freqs'
  print_rank_freqs(data, psnr_rank)


def print_rank_freqs(data, rank):
  e = len(data['examples'])
  m = len(data['models'])
  freqs = []
  for i in range(m):
    one_rank = np.count_nonzero(
        np.logical_and(np.less_equal(i + 1.0, rank), np.less(rank, i + 2.0)),
        axis=0) * 1.0 / e
    freqs.append(one_rank)
  freqs = np.array(freqs)
  print '%30s    %s' % ('', ''.join('%4.0f ' % (x + 1) for x in range(m)))
  for i, m in enumerate(data['models']):
    print '%30s    %s' % (m, ''.join(
        '%4.0f%%' % (100 * x) for x in freqs[:, i]))


def compute_diff(data):
  print '\nDIFF\n'
  # We take the first model as the best!
  ssim = np.array(data['ssim'])
  psnr = np.array(data['psnr'])
  ssim_diff = ssim - ssim[:, 0:1]
  psnr_diff = psnr - psnr[:, 0:1]
  for i, m in enumerate(data['models']):
    print '%30s    ssim-diff %.3f ± %.3f    psnr-diff %.2f ± %.2f' % (
        m, np.mean(ssim_diff[:, i]), np.std(ssim_diff[:, i]),
        np.mean(psnr_diff[:, i]), np.std(psnr_diff[:, i]))


def compute_wilcoxon(data):
  print '\nWILCOXON SIGNED-RANK TEST\n'
  # We take the first model as the basis for each comparison.
  ssim = np.array(data['ssim'])
  psnr = np.array(data['psnr'])
  for i, m in enumerate(data['models']):
    if i == 0:
      print '    [differences from %s]' % m
      continue
    ssim_v, ssim_p = wilcoxon(ssim[:, i], ssim[:, 0])
    psnr_v, psnr_p = wilcoxon(psnr[:, i], psnr[:, 0])
    print '%30s    ssim %.3f, p %.1e    psnr %.2f, p %.1e' % (m, ssim_v, ssim_p,
                                                              psnr_v, psnr_p)


def main(_):
  stats = FLAGS.stats.split(',')
  root = FLAGS.root
  model_names = FLAGS.model_names.split(',')

  data = {}
  for m in model_names:
    d = load_data(root, m)
    merge_into(data, d)

  print '\nLOADED %d models, %d examples' % (len(data['models']),
                                             len(data['examples']))

  if 'mean' in stats:
    compute_mean(data)
  if 'rank' in stats:
    compute_rank(data)
  if 'diff' in stats:
    compute_diff(data)
  if 'wilcoxon' in stats:
    compute_wilcoxon(data)
  print


if __name__ == '__main__':
  app.run()
