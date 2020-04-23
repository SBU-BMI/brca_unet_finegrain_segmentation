from util_codes import generate_polygon_json
import os, time
from multiprocessing import Pool


# run this step after while running prediction or after prediction


def generate_json_one_wsi(wsi_paths):
    wsi_path, wsi_out = wsi_paths
    print('Generating json for: ', wsi_out)
    generate_polygon_json_handler = generate_polygon_json(wsi_path, wsi_out)
    generate_polygon_json_handler.main()


def is_done_prediction(wsi_out):
    done_file = os.path.join(wsi_out, 'prediction_done.txt')
    return os.path.exists(done_file)


def is_NOT_done_gen_json(wsi_out):
    wsi_name = wsi_out.rstrip('/').split('/')[-1]
    json_fn = os.path.join(wsi_out, wsi_name + '.json')
    return not os.path.exists(json_fn)


def need_process_wsis(out_fol):
    join_path = lambda fol:os.path.join(out_fol, fol)
    fols = [fol for fol in os.listdir(out_fol) if is_done_prediction(join_path(fol)) and is_NOT_done_gen_json(join_path(fol))]
    return fols


def is_done_gen_json(out_fol):
    join_path = lambda fol:os.path.join(out_fol, fol)
    done_gen_json = [fol for fol in os.listdir(out_fol) if not is_NOT_done_gen_json(join_path(fol)) and not fol.startswith('.')]
    all_fols = [fol for fol in os.listdir(out_fol) if not fol.startswith('.')]
    return len(done_gen_json) == len(all_fols)


def main(wsi_fol, out_fol):
    while 1:
        wsi_out_fols = need_process_wsis(out_fol)
        args = [(os.path.join(wsi_fol, fol), os.path.join(out_fol, fol)) for fol in wsi_out_fols]

        if len(wsi_out_fols) == 0:
            print('No input data avail, waiting for prediction...')
            time.sleep(60)
            if is_done_gen_json(out_fol):
                return
            continue

        pool = Pool(processes=min(16, len(wsi_out_fols)))
        pool.map(generate_json_one_wsi, args)
        pool.close()


if __name__ == '__main__':
    wsi_fol = '/data01/shared/hanle/svs_tcga_seer_brca'
    out_fol = wsi_fol.rstrip('/').split('/')[-1]

    main(wsi_fol, out_fol)
