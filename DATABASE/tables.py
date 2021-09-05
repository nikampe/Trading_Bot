TABLES = {}

for coin in ['BTC', 'ETH', 'ADA']:
    TABLES['data_' + coin] = (
        "CREATE TABLE `data_" + coin + "` ("
        "  `Datetime` varchar(30) NOT NULL,"
        "  `ID` bigint NOT NULL,"
        "  `Ticker` varchar(7) NOT NULL,"
        "  `Open` double NOT NULL,"
        "  `High` double NOT NULL,"
        "  `Low` double NOT NULL,"
        "  `Close` double NOT NULL,"
        "  `Volume` bigint NOT NULL,"
        "  `5min_Return` decimal(16,4) NOT NULL,"
        "  `1d_Return` decimal(16,4) NOT NULL,"
        "  `5d_SMA` decimal(16,4) NOT NULL,"
        "  `5d_EMA` decimal(16,4) NOT NULL,"
        "  `10d_SMA` decimal(16,4) NOT NULL,"
        "  `10d_EMA` decimal(16,4) NOT NULL,"
        "  `20d_SMA` decimal(16,4) NOT NULL,"
        "  `20d_EMA` decimal(16,4) NOT NULL,"
        "  `Exchange` varchar(20) NOT NULL"
        ") ENGINE = InnoDB")

for coin in ['BTC', 'ETH', 'ADA']:
    TABLES['train_data_' + coin] = (
        "CREATE TABLE `train_data_" + coin + "` ("
        "  `Date` varchar(30) NOT NULL,"
        "  `Open` double NOT NULL,"
        "  `High` double NOT NULL,"
        "  `Low` double NOT NULL,"
        "  `Close` double NOT NULL,"
        "  `Volume` bigint NOT NULL"
        ") ENGINE = InnoDB")

TABLES['coin_tickers'] = (
    "CREATE TABLE `coin_tickers` ("
    "  `ID` int NOT NULL,"
    "  `Name` varchar(50) NOT NULL,"
    "  `Abbrev` varchar(50) NOT NULL,"
    "  `Ticker` varchar(10) NOT NULL,"
    "  PRIMARY KEY (`id`)"
    ") ENGINE = InnoDB")

TABLES['coin_portfolio'] = (
    "CREATE TABLE `coin_portfolio` ("
    "  `Datetime` varchar(30) NOT NULL,"
    "  `BTC` decimal(19,10) NOT NULL,"
    "  `BTC_Value` decimal(10,2) NOT NULL,"
    "  `BTC_Cash` decimal(10,2) NOT NULL,"
    "  `BTC_Weight` decimal(10,4) NOT NULL,"
    "  `ETH` decimal(19,10) NOT NULL,"
    "  `ETH_Value` decimal(10,2) NOT NULL,"
    "  `ETH_Cash` decimal(10,2) NOT NULL,"
    "  `ETH_Weight` decimal(10,4) NOT NULL,"
    "  `ADA` decimal(19,10) NOT NULL,"
    "  `ADA_Value` decimal(10,2) NOT NULL,"
    "  `ADA_Cash` decimal(10,2) NOT NULL,"
    "  `ADA_Weight` decimal(10,4) NOT NULL,"
    "  `Total_Value` decimal(10,2) NOT NULL"
    ") ENGINE = InnoDB")

TABLES['trades'] = (
    "CREATE TABLE `trades` ("
    "  `Date` datetime NOT NULL,"
    "  `Ticker` varchar(50) NOT NULL,"
    "  `Type` char(3) NOT NULL,"
    "  `Price` float NOT NULL,"
    "  `Amount` int NOT NULL"
    ") ENGINE = InnoDB")