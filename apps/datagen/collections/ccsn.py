import h5py
import toml

from pathlib import Path
from ccsnet.waveform import padding


ARGUMENTS_FILE = "/home/hongyin.chen/anti_gravity/CCSNet/apps/train/trainer/arguments.toml"

ccsnet_arguments = toml.load(ARGUMENTS_FILE)


def save_each_family(
    ccsn_dict,
    dur,
    sample_rate,
    sqrtnum,
    output_dir
):

    grand_dict = load_h5_as_dict(ccsn_dict=ccsn_dict)
    
    phi = np.tile(np.linspace(0, 2*np.pi, sqrtnum), sqrtnum)
    CosTheta = np.repeat(np.linspace(-1, 1, sqrtnum), sqrtnum)
    theta = np.arccos(CosTheta)
    
    for family in ccsn_dict.keys():
        
        for signal in ccsn_dict[family]:

            time = grand_dict[family][signal][0]
            quad_moment = grand_dict[family][signal][1]
            

            data = np.empty([sqrtnum**2, 2,dur*sample_rate])
            
            hp, hc = get_hp_hc_from_q2ij(quad_moment, phi, theta)
            for i in tqdm(range(sqrtnum**2)):
            
                h_c = padding(hc[i], time, dur=dur, time_shift=-0.0)
                h_p = padding(hp[i], time, dur=dur, time_shift=-0.0)
                
                data[i, 0] = h_c.value
                data[i, 1] = h_p.value
                
            prior = PriorDict()
            prior['dec'] = Cosine()
            prior['psi'] = Uniform(0, np.pi)
            prior['ra'] = Uniform(0, 2 * np.pi)

            parameter = prior.sample(sqrtnum**2)
            
            with h5py.File(output_dir / f'{family}.h5', 'a') as g:
                
                g1 = g.create_group(f"{signal}")
                g1.create_dataset('signals', data=data)
                g1.create_dataset('ori_theta', data=theta)
                g1.create_dataset('ori_phi', data=phi)
                g1.create_dataset('dec', data=parameter['dec'])
                g1.create_dataset('psi', data=parameter['psi'])
                g1.create_dataset('ra', data=parameter['ra'])

        
def merge_ccsn(
    selected_ccsn,
    data_dir,
    output_dir,
):

    data_dict = {
        "signals":[],
        "ori_theta": [],
        "ori_phi": [],
        "dec": [], 
        "psi": [],
        "ra": []
    }

    data_set = [
        "signals", 
        "ori_theta", 
        "ori_phi", 
        "dec", 
        "psi", 
        "ra"
    ]

    lenth = 0
    for family in tqdm(selected_ccsn.keys()):
        
        for signal in selected_ccsn[family]:
            
            ccsn_file = h5_thang(data_dir / f'{family}.h5')
            
            ccsn_sep_dict = ccsn_file.h5_data()
    
            for name in data_set:
                
                data_dict[f"{name}"].append(ccsn_sep_dict[f"{signal}/{name}"])
    
        lenth += len(selected_ccsn[family])*3600
    
    idx = np.linspace(0, lenth-1, lenth).astype("int")
    
    with h5py.File(output_dir / "signals.h5", "a") as g:
    
        for name in data_set:
        
            g.create_dataset(f"{name}", data=np.concatenate(data_dict[f"{name}"])[idx])
    
        g.create_dataset("shuffle_idx", data=idx)
  


if __name__ == '__main__':

    DATA_BASE = Path.home() / "Data/CCSNet_data/Publication_V3"
    DATA_BASE_RAW = DATA_BASE / "CCSN"
    DATA_BASE_RAW.mkdir(parents=True, exist_ok=True)

    CCSN_TOML = Path.home() / "Xperimental/CCSNet/sandbox/ccsn.toml"
    SELECTED_CCSN_TOML = Path.home() / "Xperimental/CCSNet/sandbox/ccsn.toml"

    save_each_family(
        ccsn_dict=toml.load(CCSN_TOML),
        dur=4,
        sample_rate=4096,
        sqrtnum=60,
        output_dir=DATA_BASE_RAW
    )