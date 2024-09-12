"""
Stopwords specific for ingredient lists
    - add_stopwords_de: common phrasing included in ingredient statements in german
    - country_list_{language}: country names
    - country_codes_{language}: two digit country codes
    - units: units of measurement
    - key_words: ingredients to keep using the bags of words methodology (abandoned)
"""

add_stopwords_de = [
    'zutaten', 'herkunft', 'produkt', 'weniger', 'schweizer', 'kann', 'spuren', 'enthalten', 'allergenhinweisekann', 'insgesamt', 'gesamtanteil', 
    'allergene', 'allergenhinweise', 'gesamtkakaobestandteile', 'gesamtmilchbestandteile', 'mind', 'mindestens', 'davon', 'insgesamt', 'beutel', 
    'durchzogen', 'sieh', 'ausgansmaterial', 'enthaltmilch', 'enthaltendunkl', 'allelandwirtschaftlich', 'hergestellt', 'ausgangsmaterial'
                ]

add_stopwords_en = [
    'ingredients', 'emulsifiers'
    ]

country_list_de = [
    'afghanistan', 'ägypten', 'albanien', 'algerien', 'andorra', 'angola', 'antigua und barbuda', 
    'äquatorialguinea', 'argentinien', 'armenien', 'aserbaidschan', 'äthiopien', 'australien', 'bahamas', 
    'bahrain', 'bangladesch', 'barbados', 'belarus', 'belgien', 'belize', 'benin', 'bhutan', 'bolivien', 
    'bosnien und herzegowina', 'botswana', 'brasilien', 'brunei', 'bulgarien', 'burkina faso', 'burundi', 
    'chile', 'volksrepublik china', 'costa rica', 'dänemark', 'deutschland', 'dominica', 'dominikanische republik',
    'dominikanische', 'dschibuti', 'ecuador', 'elfenbeinküste', 'el salvador', 'eritrea', 'estland', 
    'eswatini', 'fidschi', 'finnland', 'frankreich', 'gabun', 'gambia', 'georgien', 'ghana', 'grenada', 
    'griechenland', 'guatemala', 'guinea', 'guinea-bissau', 'guyana', 'haiti', 'honduras', 'indien', 
    'indonesien', 'irak', 'iran', 'irland', 'island', 'israel', 'italien', 'jamaika', 'japan', 'jemen', 
    'jordanien', 'kambodscha', 'kamerun', 'kanada', 'kap verde', 'kasachstan', 'katar', 'kenia', 'kirgisistan', 
    'kiribati', 'kolumbien', 'komoren', 'kongo, demokratische republik', 'kongo, republik', 'korea, nord',
    'korea, süd', 'kroatien', 'kuba', 'kuwait', 'laos', 'lesotho', 'lettland', 'libanon', 'liberia', 
    'libyen', 'liechtenstein', 'litauen', 'luxemburg', 'madagaskar', 'malawi', 'malaysia', 'malediven', 
    'mali', 'malta', 'marokko', 'marshallinseln', 'mauretanien', 'mauritius', 'mexiko', 'mikronesien', 
    'moldau', 'monaco', 'mongolei', 'montenegro', 'mosambik', 'myanmar', 'namibia', 'nauru', 'nepal', 
    'neuseeland', 'nicaragua', 'niederlande', 'niger', 'nigeria', 'nordmazedonien', 'norwegen', 'oman', 
    'österreich', 'osttimor', 'pakistan', 'palau', 'panama', 'papua-neuguinea', 'paraguay', 'peru', 'philippinen',
    'polen', 'portugal', 'republik','ruanda', 'rumänien', 'russland', 'salomonen', 'sambia', 'samoa', 
    'san marino', 'são tomé und príncipe', 'saudi-arabien', 'schweden', 'schweiz', 'senegal', 'serbien', 
    'seychellen', 'sierra leone', 'simbabwe', 'singapur', 'slowakei', 'slowenien', 'somalia', 'spanien', 
    'sri lanka', 'st. kitts und nevis', 'st. lucia', 'st. vincent und die grenadinen', 'südafrika', 'sudan',
    'südsudan', 'suriname', 'syrien', 'tadschikistan', 'tansania', 'thailand', 'togo', 'tonga', 
    'trinidad und tobago', 'tschad', 'tschechien', 'tunesien', 'türkei', 'turkmenistan', 'tuvalu', 'uganda', 
    'ukraine', 'ungarn', 'uruguay', 'usbekistan', 'vanuatu', 'venezuela', 'vereinigte arabische emirate', 
    'vereinigte staaten', 'vereinigtes königreich', 'vietnam', 'zentralafrikanische republik', 'zypern'
    ]

country_list_en = [
    'afghanistan', 'albania', 'algeria', 'andorra', 'angola', 'antigua and barbuda', 'argentina', 'armenia',
     'australia', 'austria', 'azerbaijan', 'bahamas', 'bahrain', 'bangladesh', 'barbados', 'belarus', 'belgium', 
     'belize', 'benin', 'bhutan', 'bolivia (plurinational state of)', 'bosnia and herzegovina', 'botswana', 'brazil', 
     'brunei darussalam', 'bulgaria', 'burkina faso', 'burundi', 'cabo verde', 'cambodia', 'cameroon', 'canada', 
     'central african republic', 'chad', 'chile', 'china', 'colombia', 'comoros', 'congo', 'congo, democratic republic of the', 
     'costa rica', "côte d'ivoire", 'croatia', 'cuba', 'cyprus', 'czechia', 'denmark', 'djibouti', 'dominica', 'dominican republic', 
     'ecuador', 'egypt', 'el salvador', 'equatorial guinea', 'eritrea', 'estonia', 'eswatini', 'ethiopia', 'fiji', 'finland', 
     'france', 'gabon', 'gambia', 'georgia', 'germany', 'ghana', 'greece', 'grenada', 'guatemala', 'guinea', 'guinea-bissau', 
     'guyana', 'haiti', 'honduras', 'hungary', 'iceland', 'india', 'indonesia', 'iran (islamic republic of)', 'iraq', 'ireland', 
     'israel', 'italy', 'jamaica', 'japan', 'jordan', 'kazakhstan', 'kenya', 'kiribati', "korea (democratic people's republic of)", 
     'korea, republic of', 'kuwait', 'kyrgyzstan', "lao people's democratic republic", 'latvia', 'lebanon', 'lesotho', 'liberia', 'libya', 
     'liechtenstein', 'lithuania', 'luxembourg', 'madagascar', 'malawi', 'malaysia', 'maldives', 'mali', 'malta', 'marshall islands', 
     'mauritania', 'mauritius', 'mexico', 'micronesia (federated states of)', 'moldova, republic of', 'monaco', 'mongolia', 'montenegro', 
     'morocco', 'mozambique', 'myanmar', 'namibia', 'nauru', 'nepal', 'netherlands', 'new zealand', 'nicaragua', 'niger', 'nigeria', 
     'north macedonia', 'norway', 'oman', 'pakistan', 'palau', 'panama', 'papua new guinea', 'paraguay', 'peru', 'philippines', 
     'poland', 'portugal', 'qatar', 'romania', 'russian federation', 'rwanda', 'saint kitts and nevis', 'saint lucia', 
     'saint vincent and the grenadines', 'samoa', 'san marino', 'sao tome and principe', 'saudi arabia', 'senegal', 
     'serbia', 'seychelles', 'sierra leone', 'singapore', 'slovakia', 'slovenia', 'solomon islands', 'somalia', 'south africa', 
     'south sudan', 'spain', 'sri lanka', 'sudan', 'suriname', 'sweden', 'switzerland', 'syrian arab republic', 'tajikistan', 
     'tanzania, united republic of', 'thailand', 'timor-leste', 'togo', 'tonga', 'trinidad and tobago', 'tunisia', 'türkiye', 
     'turkmenistan', 'tuvalu', 'uganda', 'ukraine', 'united arab emirates', 'united kingdom of great britain and northern ireland', 
     'united states of america', 'uruguay', 'uzbekistan', 'vanuatu', 'venezuela (bolivarian republic of)', 'viet nam', 'yemen', 'zambia', 'zimbabwe'
     ]

country_codes_de = [
    'af', 'eg', 'al', 'dz', 'ad', 'ao', 'ag', 'gq', 'ar', 'am', 'az', 'et', 'au', 'bs', 'bh', 'bd', 'bb', 'by', 'be', 'bz', 'bj', 
    'bt', 'bo', 'ba', 'bw', 'br', 'bn', 'bg', 'bf', 'bi', 'cl', 'cn', 'cr', 'dk', 'de', 'dm', 'do', 'dj', 'ec', 'ci', 'sv', 'er', 
    'ee', 'sz', 'fj', 'fi', 'fr', 'ga', 'gm', 'ge', 'gh', 'gd', 'gr', 'gt', 'gn', 'gw', 'gy', 'ht', 'hn', 'in', 'id', 'iq', 'ir', 
    'ie', 'is', 'il', 'it', 'jm', 'jp', 'ye', 'jo', 'kh', 'cm', 'ca', 'cv', 'kz', 'qa', 'ke', 'kg', 'ki', 'co', 'km', 'cd', 'cg', 
    'kp', 'kr', 'hr', 'cu', 'kw', 'la', 'ls', 'lv', 'lb', 'lr', 'ly', 'li', 'lt', 'lu', 'mg', 'mw', 'my', 'mv', 'ml', 'mt', 'ma', 
    'mh', 'mr', 'mu', 'mx', 'fm', 'md', 'mc', 'mn', 'me', 'mz', 'mm', 'na', 'nr', 'np', 'nz', 'ni', 'nl', 'ne', 'ng', 'mk', 'no', 
    'om', 'at', 'tl', 'pk', 'pw', 'pa', 'pg', 'py', 'pe', 'ph', 'pl', 'pt', 'rw', 'ro', 'ru', 'sb', 'zm', 'ws', 'sm', 'st', 'sa', 
    'se', 'ch', 'sn', 'rs', 'sc', 'sl', 'zw', 'sg', 'sk', 'si', 'so', 'es', 'lk', 'kn', 'lc', 'vc', 'za', 'sd', 'ss', 'sr', 'sy', 
    'tj', 'tz', 'th', 'tg', 'to', 'tt', 'td', 'cz', 'tn', 'tr', 'tm', 'tv', 'ug', 'ua', 'hu', 'uy', 'uz', 'vu', 've', 'ae', 'us', 
    'gb', 'vn', 'cf', 'cy'
    ]

country_codes_en = [
    'af', 'al', 'dz', 'ad', 'ao', 'ag', 'ar', 'am', 'au', 'at', 'az', 'bs', 'bh', 'bd', 'bb', 'by', 'be', 'bz', 'bj', 'bt', 'bo', 
    'ba', 'bw', 'br', 'bn', 'bg', 'bf', 'bi', 'cv', 'kh', 'cm', 'ca', 'cf', 'td', 'cl', 'cn', 'co', 'km', 'cg', 'cd', 'cr', 'ci', 
    'hr', 'cu', 'cy', 'cz', 'dk', 'dj', 'dm', 'do', 'ec', 'eg', 'sv', 'gq', 'er', 'ee', 'sz', 'et', 'fj', 'fi', 'fr', 'ga', 'gm', 
    'ge', 'de', 'gh', 'gr', 'gd', 'gt', 'gn', 'gw', 'gy', 'ht', 'hn', 'hu', 'is', 'in', 'id', 'ir', 'iq', 'ie', 'il', 'it', 'jm', 
    'jp', 'jo', 'kz', 'ke', 'ki', 'kp', 'kr', 'kw', 'kg', 'la', 'lv', 'lb', 'ls', 'lr', 'ly', 'li', 'lt', 'lu', 'mg', 'mw', 'my', 
    'mv', 'ml', 'mt', 'mh', 'mr', 'mu', 'mx', 'fm', 'md', 'mc', 'mn', 'me', 'ma', 'mz', 'mm', 'na', 'nr', 'np', 'nl', 'nz', 'ni', 
    'ne', 'ng', 'mk', 'no', 'om', 'pk', 'pw', 'pa', 'pg', 'py', 'pe', 'ph', 'pl', 'pt', 'qa', 'ro', 'ru', 'rw', 'kn', 'lc', 'vc', 
    'ws', 'sm', 'st', 'sa', 'sn', 'rs', 'sc', 'sl', 'sg', 'sk', 'si', 'sb', 'so', 'za', 'ss', 'es', 'lk', 'sd', 'sr', 'se', 'ch', 
    'sy', 'tj', 'tz', 'th', 'tl', 'tg', 'to', 'tt', 'tn', 'tr', 'tm', 'tv', 'ug', 'ua', 'ae', 'gb', 'us', 'uy', 'uz', 'vu', 've', 
    'vn', 'ye', 'zm', 'zw'
]

units = [
    'kj', 'kcal', 'kg', 'g', 'mg', 'pro', 'mgl', 'l', 'µg', 'mgkg'
    ]

key_words = [

]

grouped_words_de = word_list ={
    'acerola': ['acerolafruchtpulv', 'acerolaextrak', 'acerolapulv'],
    'acesulfam': ['acesulfamek', 'acesulfamk', 'acesultamk'],
    'ammoniumcarbonat': ['ammoniumcarbonate', 'ammoniumhydrogencarbonat'],
    'anana': ['ananasstuck'],
    'antioxida': [ 'antioxidatie', 'antioxidationsm', 'antioxidationsmittel', 'antioxydationsmittel'],
    'apfelarom': ['apfelaroma'],
    'arom': ['aromaextrakt', 'aromastoff', 'aroma'],
    'artischock': ['artischockenherz'],
    'ascorbi': ['ascorbinsaur', 'ascorbic'],
    'backmittel': ['backtreibmittel', 'backtriebmittel'],
    'baen': ['baenstuckch'],
    'balsamico': ['balsamicoessig'],
    'basilikum': ['basil'],
    'baumnuss': ['baumnüsse'],
    'baumwollsaatol': ['baumwollsamenol'],
    'beer': ['beerenmischung'],
    'butt': ['beurr'],
    'bifidobacterium': ['bifidusbakterie', 'bifiduskultur', ],
    'bio': ['biologisch', 'biologiqu', 'biologischerlandwirtschaf', 'bioproduktio', 'biozertifizier', 'biozertifizierung'],
    'borlotti': ['borlottiboh'],
    'broccoli': ['brokkoli'], 
    'cashew': ['cashewk', 'cashewnuss', 'cashewstuck'],
    'cayennepfeff': ['cayennepfefferpulv'],
    'cellulo': ['cellulosegummi'],
    'chili': ['chilli'],
    'citronensaur': ['citronsaur'],
    'cranberrie': ['cranberry', 'cranberryzubereitung'],
    'trock': ['dried'],
    'cream': ['crem'],
    'wass': ['eau', 'mineralwass', 'wasser', 'trinkwass'],
    'eierspatzl': ['eierspatzli'],
    'eisberg': ['eisbergsala'],
    'endivie': ['endiviensala'],
    'erdbeerenzubereitung': ['erdbeerzubereitung'],
    'fruktosesirup': ['fructosesirup'],
    'frukto': ['fructo', 'fruchtzuck'],
    'fruktoseglukosesirup': ['fructoseglucosesirup', 'glukosefruktosesirup', 'glucosefructosesirup' ],
    'gluko': ['gluco'],
    'glukosesirup': {'glucosesirup'},
    'granatapfelsaf': ['grapefruitsaf'],
    'hydrolisier': ['hydrolysier'],
    'iodier': ['iod', 'iodized', 'jodier'],
    'joghur': ['jogur', 'vollmilchjoghur', 'vollmilchjogur', 'magerjoghur'],
    'kas': ['bergmagerka', 'halbhartka', 'hartka', 'gruyèr', 'emmental'],
    'kakao': ['cacao'],
    'karamellsirup': ['caramelsirup'],
    'karamellzuck': ['caramelzuck'],
    'koffei': ['caffei'],
    'salz': ['kochsalz', 'meersalz', 'mineralsalz', 'nacl', 'speisesalz'],
    'salzgehal': ['kochsalzgehal'],
    'kohlendioxid': ['kohlendioxyd'],
    'kokosfett': ['kokosnussfett'],
    'kokosmilch': ['kokosnussmilch'],
    'kokoswass': ['kokosnusswass'],
    'crevett': ['krevett'],
    'lach': ['seelach', 'rauchlach', 'wildlach'],
    'lakto': ['lacto'],
    'limettenarom': ['limettenaroma'],
    'mango': ['mangostuck', 'mangowurfel'],
    'milch': ['kuhmilch', 'schafmilch', 'rohmilch', 'vollmilch', 'biovollmilch', 
              'magermilch', 'bergmilch', 'buffelmilch', 'milk'],
    'natriumcarbona': ['natriumkarbona'],
    'ol': ['oil', 'oel'],
    'oligofructo': ['oligofrukto'],
    'pasteurisier': ['pasteutisier', 'pas'],
    'peperoncino': ['peperoncini'],
    'rahm': ['vollrahm', 'biorahm', 'doppelrahm', 'schlagrahm' ],
    'salat': ['kopfsala'],
    'schokolad': ['milchschokolad', 'vollmilchschokolad'],
    'schink': [ 'rohschink', 'hinterschink', 'vorderschink', 'schweineschink'],
    'senfsam': [ 'senfkor', ],
    'sonnenblumenol': ['sonnennblumenol', 'sonenblumenol'],
    'sorbinsaaur': ['sorbinsaur'],
    'essig': ['tafelessig'],
    'tapioca': ['tapioka'],
    'waffel': ['waffelstuckch'],
    'weizenmehl': ['weizenemehl'],
    'wurzmischung': ['wurzemischung', 'wurzzubereitung'],
    'zucchetti': ['zucchini'],
    'zuck': ['sugar'],
    'citronensaur': ['zitronensaur'],
    'vanilleextrak': ['vanilli', 'vanillearoma', 'vanill', 'bourbonvanill' ],
    'weinessig': ['branntweinessig', 'weissweinessig', 'rotweinessig'],
    'rohrzuck': ['rohrohrzuck', ],
    'paprika': ['paprikaextrak'],
    'milchpulv': ['vollmilchpulv', 'magermilchpulv',],
    'kaffee': ['rostkaffee'],
    'margari': ['pflanzenmargari'],
    'frischka': ['rahmfrischka', 'doppelrahmfrischka', 'magerfrischka', 'bergfrischka'],
    'tomatenmark': ['tomatenpuree'],
    'rahmpulv': [ 'sahnepulv'],
    'apfel': ['apfelstuck'],
    'caramel': ['karamell'],
    'vollkornhaferflock': ['hafervollkornflock'],
    'weizenvollkornmehl': ['vollkornweizenmehl']
    }