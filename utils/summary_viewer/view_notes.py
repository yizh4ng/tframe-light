import sys, os
FILE_PATH = os.path.abspath(__file__)
ROOT = FILE_PATH
# Specify the directory depth with respect to the root of your project here
# (The project root usually holds your data folder and has a depth of 0)
DIR_DEPTH = 1
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
from tframe.utils.summary_viewer import main_frame
from tframe import local


current_dir = os.path.dirname(FILE_PATH)
if len(sys.argv) == 2:
  path = sys.argv[1]
  if os.path.exists(path):
    current_dir = path


default_inactive_flags = (
  # 'patience',
  'shuffle',
  'epoch',
  'early_stop',
  'warm_up_thres',
  'mark',
  # 'batch_size',
  'save_mode',
  'output_size',
  'total_params',
  'num_units',
  'weights_fraction',
)
default_inactive_criteria = (
  'Mean Record',
  # 'Record',
  'SC(LCS)',
  'LC(SCS)',
  'Best Accuracy',
  'Best F1',
)
flags_to_ignore = (
  'lr',
  'use_conveyor',
  'hyper_kernel',
  'horizon',
  'global_lr_penalty',
  'use_lob_sig_curve',
  'mark',
  # 'patience',
  'shuffle',
  'warm_up_thres',
  # 'batch_size',
  'epoch',
  'early_stop',
  # 'num_steps',
  'early_stop_on_loss',
  'clip_method',
  'data_config',
  'radical_penalty',
  'base_weight',
  'beta',
  'epsilon',
  'optimizer',
  'use_xtx_gen_rnn_batches',
  'use_wheel',
  'use_sig_curve',
  'tanh_coef',
  'log_size',
  'loss_string',
  'train_complete_set',
  'xtx_sample_len_pct',
  'all_in_one',
  # 'hidden_dim',
  # 'hyper_kernel',
  # 'input_dropout',
  'clip_threshold',
  'early_stop_on_cscore',
  'address_bias',
  'decide_on_probe',
  'head_bias',
  'address_bias',
)

while True:
  try:
    summ_path = local.wizard(extension='sum', max_depth=3,
                             current_dir=current_dir,
                             # input_with_enter=True)
                             input_with_enter=False)

    if summ_path is None:
      input()
      continue
    print('>> Loading notes, please wait ...')
    viewer = main_frame.SummaryViewer(
      summ_path,
      default_inactive_flags=default_inactive_flags,
      default_inactive_criteria=default_inactive_criteria,
      flags_to_ignore=flags_to_ignore,
    )
    viewer.show()

  except Exception as e:
    import sys, traceback
    traceback.print_exc(file=sys.stdout)
    input('Press any key to quit ...')
    raise e
  print('-' * 80)
