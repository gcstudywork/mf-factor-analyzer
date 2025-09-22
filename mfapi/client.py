import requests
import time
from datetime import datetime

MFAPI_BASE_URL = "https://api.mfapi.in"

class MFAPI:
    def __init__(self, cache_duration=3600):
        self.base_url = MFAPI_BASE_URL
        self.scheme_codes_dict = None
        self.last_fetch_time = None
        self.cache_duration = cache_duration
    
    def get_scheme_codes(self):
        """Get all scheme codes with caching"""
        current_time = time.time()
        
        if (self.scheme_codes_dict is None or 
            self.last_fetch_time is None or 
            (current_time - self.last_fetch_time) > self.cache_duration):
            
            response = requests.get(f"{self.base_url}/mf")
            response.raise_for_status()
            schemes_data = response.json()
            
            self.scheme_codes_dict = {
                str(scheme['schemeCode']): scheme['schemeName'] 
                for scheme in schemes_data
            }
            self.last_fetch_time = current_time
        
        return self.scheme_codes_dict
    
    def get_scheme_details(self, scheme_code):
        """Get basic scheme details"""
        response = requests.get(f"{self.base_url}/mf/{scheme_code}")
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'SUCCESS':
            meta = data.get('meta', {})
            latest_nav = data.get('data', [{}])[0].get('nav', 'N/A') if data.get('data') else 'N/A'
            
            return {
                'scheme_name': meta.get('scheme_name', 'N/A'),
                'scheme_code': meta.get('scheme_code', 'N/A'),
                'fund_house': meta.get('fund_house', 'N/A'),
                'scheme_type': meta.get('scheme_type', 'N/A'),
                'scheme_category': meta.get('scheme_category', 'N/A'),
                'nav': latest_nav
            }
        return None
    
    def get_scheme_historical_nav(self, scheme_code, from_date=None, to_date=None):
        """Get historical NAV data for a scheme"""
        response = requests.get(f"{self.base_url}/mf/{scheme_code}")
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'SUCCESS':
            nav_data = data.get('data', [])
            
            if from_date or to_date:
                filtered_data = []
                from_dt = datetime.strptime(from_date, '%Y-%m-%d') if from_date else None
                to_dt = datetime.strptime(to_date, '%Y-%m-%d') if to_date else None
                
                for entry in nav_data:
                    try:
                        entry_date = datetime.strptime(entry['date'], '%d-%m-%Y')
                        
                        if from_dt and entry_date < from_dt:
                            continue
                        if to_dt and entry_date > to_dt:
                            continue
                        
                        filtered_data.append(entry)
                    except (ValueError, KeyError):
                        continue
                
                return filtered_data
            
            return nav_data
        return None