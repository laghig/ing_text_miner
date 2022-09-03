"""
Additional Stopwords specific for ingredient lists
    - add_stopwords_de: common phrasing included in ingredient statements
    - country_list_de: country names in german
    - country_codes_de: two digit country codes
    - units: units of measurement
"""

add_stopwords_de = ['zutaten', 'herkunft', 'produkt', 'weniger', 'schweizer', 'kann', 'spuren', 'enthalten', 'allergenhinweisekann', 'insgesamt', 'gesamtanteil', 
                'allergene', 'allergenhinweise', 'gesamtkakaobestandteile', 'gesamtmilchbestandteile', 'mind', 'mindestens', 'davon', 'insgesamt', 'beutel', 
                'durchzogen', 'sieh', 'ausgansmaterial', 'enthaltmilch', 'enthaltendunkl', 'allelandwirtschaftlich', 'hergestellt', 'ausgangsmaterial'
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

country_codes_de = [
    'ch', 'de', 'nl', 'fr' 
    ]

units = [
    'kj', 'kcal', 'kg', 'g', 'mg', 'pro', 'mgl', 'l', 'µg', 'mgkg'
    ]