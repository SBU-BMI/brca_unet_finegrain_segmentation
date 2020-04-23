import os, sys, time, random
import traceback
from prediction_WSIs_single_machine import run_prediction_WSIs


def touch_file(fn):
    os.system('touch ' + fn)


def rm_file(fn):
    os.system('rm -f ' + fn)


def rm_folder(fol):
    os.system('rm -rf ' + fol)


def create_fol(fol):
    if not os.path.exists(fol):
        os.mkdir(fol)


def clean_files(fol, limit_time=60):
    files = {f for f in os.listdir(fol) if not f.startswith('.')}
    for fn in files:
        fn_path = os.path.join(fol, fn)
        (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(fn_path)
        if (time.time() - ctime) / 60 > limit_time:  # have been created more than 2hrs ago
            rm_file(os.path.join(fol, fn))


def list_files(fol, template=''):
    files = [f for f in os.listdir(fol) if not f.startswith('.') and template in f]
    return files


def is_path_exists(*args):
    path = '/'.join(args)
    return os.path.exists(path)


def clean_done_fol(done_fol, out_fol, indicator_file='prediction_done.txt'):
    done_fns = list_files(done_fol)
    out_fns = list_files(out_fol)
    for fn in done_fns:
        if not is_path_exists(out_fol, fn, indicator_file):
            rm_file(os.path.join(done_fol, fn))
    for fn in out_fns:
        if is_path_exists(out_fol, fn, indicator_file) and not is_path_exists(done_fol, fn):
            touch_file(os.path.join(done_fol, fn))


if __name__ == '__main__':
    # parameter to change =======================================================
    IN_FOLDER = '/data01/shared/hanle/svs_tcga_seer_brca'
    wsi_extension = 'svs'
    model_path = 'model/CP1860_resolution10_APS448_Test_upLearned_best_0.8005.pth'
    OUT_FOLDER = IN_FOLDER.rstrip('/').split('/')[-1]
    # end of parameter to change ================================================

    create_fol(OUT_FOLDER)
    run_prediction_WSIs_handler = run_prediction_WSIs(IN_FOLDER, wsi_extension, OUT_FOLDER, model_path)

    indicator_file = 'prediction_done.txt'
    done_fol = 'done'
    processing_fol = 'processing'

    create_fol(done_fol)
    create_fol(processing_fol)

    clean_done_fol(done_fol, OUT_FOLDER, indicator_file)

    time.sleep(random.randint(100, 1000) / 100.0)  # wait for 1 --> 10s to avoid concurrency
    start_time = time.time()
    while (1):
        svs_done = set(list_files(done_fol))
        svs_processing = set(list_files(processing_fol))
        svs_all = set(list_files(IN_FOLDER, wsi_extension))
        svs_remaining = svs_all.difference(svs_done.union(svs_processing))
        if len(svs_remaining) == 0:
            exit(0)

        clean_files(processing_fol, 60)

        svs_remaining = list(svs_remaining)
        random.shuffle(svs_remaining)
        slide_name = svs_remaining[0]
        slide_path = os.path.join(IN_FOLDER, slide_name)
        try:
            touch_file(os.path.join(processing_fol, slide_name))
            run_prediction_WSIs_handler.run_prediction_one_WSI(slide_path)
        except:
            print("Fail predicting slide: ", slide_path)
            traceback.print_exc(file=sys.stdout)

        touch_file(os.path.join(done_fol, slide_name))
        rm_file(os.path.join(processing_fol, slide_name))
