"""
TCNM/data/loader_training.py - Data loader for training
Handles Vietnam TC data structure
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TCTrainingDataset(Dataset):
    """
    TC Dataset for training
    Handles normalized data for training
    """
    def __init__(self, data_root, obs_len=8, pred_len=4):
        self.data_root = data_root
        self.obs_len = obs_len
        self.pred_len = pred_len
        
        self.sequences = []
        self._load_data()
    
    def _load_data(self):
        """Load and normalize training data"""
        # data1d_dir = os.path.join(self.data_root, 'Data1d')
        data1d_dir = self.data_root 
        
        if not os.path.exists(data1d_dir):
            print(f"Warning: {data1d_dir} not found")
            return
        
        txt_files = [f for f in os.listdir(data1d_dir) if f.endswith('.txt')]
        
        for txt_file in txt_files:
            year_str = txt_file.split('_')[0]
            self._process_file(os.path.join(data1d_dir, txt_file), year_str)
        
        print(f"Loaded {len(self.sequences)} training sequences")
    
    def _normalize_data(self, lon, lat, pres, wind):
        """Normalize according to paper's formula"""
        # LONG normalization: (LONG - 1800) / 50
        lon_norm = (lon - 1800.0) / 50.0
        
        # LAT normalization: LAT / 50
        lat_norm = lat / 50.0
        
        # PRES normalization: (PRES - 960) / 50
        pres_norm = (pres - 960.0) / 50.0
        
        # WND normalization: (WND - 40) / 25
        wind_norm = (wind - 40.0) / 25.0
        
        return lon_norm, lat_norm, pres_norm, wind_norm
    
    def _process_file(self, file_path, year):
        """Process training file with normalization"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            storm_name = None
            tc_data = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                try:
                    timestamp = parts[0]
                    lon = float(parts[1])
                    lat = float(parts[2])
                    pres = float(parts[3])
                    wind = float(parts[4])
                    
                    if storm_name is None and len(parts) > 5:
                        storm_name = parts[5]
                    
                    # Normalize
                    lon_norm, lat_norm, pres_norm, wind_norm = self._normalize_data(
                        lon, lat, pres, wind
                    )
                    
                    tc_data.append({
                        'timestamp': timestamp,
                        'lon': lon_norm,
                        'lat': lat_norm,
                        'pres': pres_norm,
                        'wind': wind_norm,
                        'year': year,
                        'storm_name': storm_name or 'UNKNOWN'
                    })
                except (ValueError, IndexError):
                    continue
            
            # Create sequences
            total_len = self.obs_len + self.pred_len
            for i in range(len(tc_data) - total_len + 1):
                seq = tc_data[i:i + total_len]
                self.sequences.append(seq)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        obs_seq = seq[:self.obs_len]
        pred_seq = seq[self.obs_len:]
        
        # Extract normalized features
        obs_traj = torch.tensor([[d['lon'], d['lat']] for d in obs_seq], dtype=torch.float32)
        pred_traj = torch.tensor([[d['lon'], d['lat']] for d in pred_seq], dtype=torch.float32)
        
        obs_Me = torch.tensor([[d['pres'], d['wind']] for d in obs_seq], dtype=torch.float32)
        pred_Me = torch.tensor([[d['pres'], d['wind']] for d in pred_seq], dtype=torch.float32)
        
        # Load 3D and env data
        year = obs_seq[0]['year']
        storm_name = obs_seq[0]['storm_name']
        
        image_obs = self._load_3d_data(year, storm_name, obs_seq)
        env_data = self._load_env_data(year, storm_name, obs_seq)
        
        return (
            obs_traj, pred_traj, None, None, None, None, None,
            obs_Me, pred_Me, None, None, image_obs, None, env_data
        )
    
    def _load_3d_data(self, year, storm_name, obs_seq):
        """Load 3D data"""
        try:
            data3d_dir = os.path.join(self.data_root.replace('Data1d', 'Data3d'), 
                                     year, storm_name)
            
            images = []
            for obs in obs_seq:
                timestamp = obs['timestamp']
                npy_path = os.path.join(data3d_dir, f"{timestamp}.npy")
                
                if os.path.exists(npy_path):
                    img = np.load(npy_path)
                    images.append(img)
                else:
                    images.append(np.zeros((64, 64)))
            
            images = np.stack(images)
            return torch.from_numpy(images).float().unsqueeze(0).unsqueeze(0)
        
        except:
            return torch.zeros(1, 1, self.obs_len, 64, 64)
    
    def _load_env_data(self, year, storm_name, obs_seq):
        """Load environment data"""
        try:
            env_dir = os.path.join(self.data_root.replace('Data1d', 'Env'), 
                                  year, storm_name)
            
            env_data = {
                'wind': torch.zeros(self.obs_len, 1),
                'month': torch.zeros(self.obs_len, 12),
                'move_velocity': torch.zeros(self.obs_len, 1)
            }
            
            for i, obs in enumerate(obs_seq):
                timestamp = obs['timestamp']
                npy_path = os.path.join(env_dir, f"{timestamp}.npy")
                
                if os.path.exists(npy_path):
                    env = np.load(npy_path, allow_pickle=True).item()
                    
                    if 'wind' in env:
                        env_data['wind'][i] = torch.tensor([env['wind']])
                    if 'month' in env:
                        month_onehot = torch.zeros(12)
                        month_onehot[env['month'] - 1] = 1
                        env_data['month'][i] = month_onehot
            
            return env_data
        
        except:
            return {
                'wind': torch.zeros(self.obs_len, 1),
                'month': torch.zeros(self.obs_len, 12),
                'move_velocity': torch.zeros(self.obs_len, 1)
            }


def seq_collate_training(data):
    """Collate for training"""
    batch_size = len(data)
    
    obs_traj_list = []
    pred_traj_list = []
    obs_Me_list = []
    pred_Me_list = []
    image_obs_list = []
    env_data_list = []
    
    for item in data:
        obs_traj_list.append(item[0])
        pred_traj_list.append(item[1])
        obs_Me_list.append(item[7])
        pred_Me_list.append(item[8])
        image_obs_list.append(item[11])
        env_data_list.append(item[13])
    
    obs_traj = torch.stack(obs_traj_list, dim=1)
    pred_traj = torch.stack(pred_traj_list, dim=1)
    obs_Me = torch.stack(obs_Me_list, dim=1)
    pred_Me = torch.stack(pred_Me_list, dim=1)
    
    image_obs = torch.cat(image_obs_list, dim=0)
    
    env_data = {}
    for key in env_data_list[0].keys():
        env_data[key] = torch.stack([d[key] for d in env_data_list], dim=0)
    
    return [
        obs_traj,      # 0
        pred_traj,     # 1
        None,          # 2
        None,          # 3
        None,          # 4
        None,          # 5
        None,          # 6
        obs_Me,        # 7
        pred_Me,       # 8
        None,          # 9
        None,          # 10
        image_obs,     # 11
        None,          # 12
        None,          # 13
        None,          # 14
        env_data       # 15 <--- Đưa về index 15
    ]


def data_loader(args, path, test=False, batch_size=None):
    """Create training data loader"""
    dataset = TCTrainingDataset(
        data_root=path['root'],
        obs_len=args.obs_len,
        pred_len=args.pred_len
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size or args.batch_size,
        shuffle=not test,
        collate_fn=seq_collate_training,
        num_workers=4,
        pin_memory=True, 
        drop_last=True
    )
    
    return dataset, loader