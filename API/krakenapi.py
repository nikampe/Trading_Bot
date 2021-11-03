import requests
import urllib.parse
import hashlib
import hmac
import base64
import time

API_URL = "https://api.kraken.com"
API_KEY_PUBL = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
API_KEY_PRIV = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

class CoinAPI:
    def __init__(self):
        self.api_url = API_URL
        self.priv_key = API_KEY_PRIV
        self.publ_key = API_KEY_PUBL
    def getKrakenSignature(self, urlpath, data, secret):
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()
    def getRequest(self, uri_path, data):
        headers = {}
        headers['API-Key'] = self.publ_key
        headers['API-Sign'] = self.getKrakenSignature(uri_path, data, self.priv_key)             
        req = requests.post((self.api_url + uri_path), headers = headers, data = data)
        return req
    def getBalance(self):
        resp = self.getRequest('/0/private/Balance', {"nonce": str(int(1000*time.time()))})
        account_balance = resp.json()
        return account_balance
    def getTrades(self):
        resp = self.getRequest('/0/private/TradesHistory', {"nonce": str(int(1000*time.time())), "trades": True})
        trades = resp.json()
        return trades
    def placeBuyOrder(self, pair, volume):
        resp = self.getRequest('/0/private/AddOrder', {"nonce": str(int(1000*time.time())), "ordertype": "market", "type": "buy", "volume": volume, "pair": pair})
        buy_order = resp.json()
        return buy_order
    def placeSellOrder(self, pair, volume):
        resp = self.getRequest('/0/private/AddOrder', {"nonce": str(int(1000*time.time())), "ordertype": "market", "type": "sell", "volume": volume, "pair": pair})
        sell_order = resp.json()
        return sell_order
    def cancelOrders(self):
        resp = self.getRequest('/0/private/CancelAll', {"nonce": str(int(1000*time.time()))})
        canceled_orders = resp.json()
        return canceled_orders

if __name__ == '__main__':
    CoinAPI().getBalance()