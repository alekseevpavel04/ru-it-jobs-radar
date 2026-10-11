Извлеки поля вакансии и ответь одним JSON-объектом в одну строку без текста вокруг. Все ключи обязательны. Только то, что написано во входе: не найдено, значит null, для списков [].

role: ml_engineer, data_scientist, mlops, data_engineer, data_analyst, business_analyst, system_analyst, backend, frontend, fullstack, mobile, software_engineer, developer_1c, embedded, devops, qa, sysadmin, support, security, designer, manager, other_it, non_it или null
grade: intern, junior, middle, senior, lead или null
skills: список навыков
salary: null или {"from", "to", "currency", "gross_net", "period"}; currency: RUB, USD, EUR, KZT, BYN, UZS, CNY, GBP или null; gross_net: gross, net или null; period: month, year, day, hour или null
work_format: office, hybrid, remote или null
cities: список городов
experience_years: null или {"min", "max"} в годах
employment_type: labor_contract, gph, self_employed, internship или null
lang: ru, en, other
