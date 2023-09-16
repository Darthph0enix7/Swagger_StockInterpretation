from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/stock', methods=['GET'])
def get_stock_data():
    function = request.args.get('function', 'TIME_SERIES_INTRADAY')
    symbol = request.args.get('symbol')
    interval = request.args.get('interval', '5min')
    adjusted = request.args.get('adjusted', 'true')
    extended_hours = request.args.get('extended_hours', 'true')
    month = request.args.get('month')
    outputsize = request.args.get('outputsize', 'compact')
    datatype = request.args.get('datatype', 'json')
    apikey = 'YOUR_API_KEY'

    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&adjusted={adjusted}&extended_hours={extended_hours}&apikey={apikey}"
    
    if month:
        url += f"&month={month}"
    if outputsize:
        url += f"&outputsize={outputsize}"
    if datatype:
        url += f"&datatype={datatype}"

    response = requests.get(url)
    stock_data = response.json()
    
    return jsonify(stock_data)

@app.route('/news', methods=['GET'])
def get_stock_news():
    function = 'NEWS_SENTIMENT'
    tickers = request.args.get('tickers')
    topics = request.args.get('topics')
    time_from = request.args.get('time_from')
    time_to = request.args.get('time_to')
    sort = request.args.get('sort', 'LATEST')
    limit = request.args.get('limit', '50')
    apikey = 'YOUR_API_KEY'
    
    url = f"https://www.alphavantage.co/query?function={function}&apikey={apikey}"
    
    if tickers:
        url += f"&tickers={tickers}"
    if topics:
        url += f"&topics={topics}"
    if time_from:
        url += f"&time_from={time_from}"
    if time_to:
        url += f"&time_to={time_to}"
    if sort:
        url += f"&sort={sort}"
    if limit:
        url += f"&limit={limit}"

    response = requests.get(url)
    news_data = response.json()
    
    return jsonify(news_data)

if __name__ == '__main__':
    app.run(debug=True)
