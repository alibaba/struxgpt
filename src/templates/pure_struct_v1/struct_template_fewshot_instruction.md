You are a helpful NLP assistant. I am working on constructing an automatic language assessment protocol with structural analysis.

Help me rephrase a given statement to identify:

(1) a summary of the statement's scope, which should generally be a noun phrase with a few words.

(2) a list of main aspects on which the statement is discussing. Each main aspect should be a noun or noun phrase. The aspect list should be precise and concise to conclude the statement, and the number of aspects should be limited.

(3) an enumeration of descriptive sentences regarding each aspect above, which display the details of those aspects. Each description sentence must be completed and faithful to the original statement. You shoud NOT remove any descriptive segment in this layer.

Given an original statement, the rephrased structure should strictly follow this two-layer format:

## Statement's scope:
```[generally a noun phrase]```

## Statement's main aspects and corresponding descriptions:
```
1. [the first aspect of the statement]
    1.1. [a descriptive sentence corresponding to this aspect]
    1.2. [a descriptive sentence corresponding to this aspect]
    1.3. [another descriptive sentence, if necessary]
2. [the second aspect of the statement]
    2.1. [a descriptive sentence corresponding to this aspect]
    2.2. [another descriptive sentence, if necessary]
3. [another aspect of the statement, if necessary]
```


Here is an example to illustrate how to rephrase an input statement as the desired structure. Pay attention to numerical or enumeration indicators, like '1. ', '2. ', '(1) ', '(2) ', '- ', etc.

# Input:
```
Comprehensive prevention measures for malignant diseases in the rice seedling stage are as follows:
1. Choose disease-free seeds. Do not leave seeds in diseased fields and nearby rice fields. Choose healthy rice and eliminate diseased, dead, and injured rice.
2. Seed disinfection. Before sowing, soak the seeds with 25% 100g (Xibok) EC 3000 times liquid for 1 to 2 days, or take 20 grams of 17% Dexinqing wettable powder for every 6 kilograms of rice seeds. Soak the seeds in 8 kg of water for 60 hours.
3. Deal with diseased rice straw. Do not cover germinated or dry seedlings with diseased straw.
```

# Output:

## Statement's scope:
```The comprehensive prevention measures for malignant diseases in rice seedling```

## Statement's main aspects and corresponding descriptions:
```
1. Choose disease-free seeds
    1.1. Do not leave seeds in diseased fields and nearby rice fields.
    1.2. Choose healthy rice and eliminate diseased, dead, and injured rice.
2. Seed disinfection
    2.1. Before sowing, soak the seeds with 25% 100g (Xibok) EC 3000 times liquid for 1 to 2 days.
    2.2. An altinative is taking 20 grams of 17% Dexinqing wettable powder for every 6 kilograms of rice seeds.
    2.3. Soak the seeds in 8 kg of water for 60 hours.
3. Deal with diseased rice straw
    3.1. Do not cover germinated or dry seedlings with diseased straw.
```

Here is another example to illustrate how to rephrase an input statement as the desired structure. When there are no explicit words indicating the statement's scope and main aspects, please use a few words to precisely summarize the scope as well as the main aspects. Then you may carefully attach the descriptive sentences to each main aspect.

# Input: 
```
The water absorption curve of rice after soaking is a unimodal curve. The inflection point of the curve is the peak period of water absorption. The relationship between rice water absorption and time is non-linear and can be expressed by the following formula: a*t+b=c, where a, b, and c are constants. Under different humidity conditions, the change in the water absorption rate of rice with time is basically similar, that is, the water absorption rate is between 0 and the point d inflection, and the rate of change accelerates over time. After the inflection point, the increase in water absorption gradually stabilizes. The changing rules of water absorption and moisture content of rice are similar, but under different humidity conditions, the relationship between water absorption of rice and moisture content is different. When the moisture content is low, the water absorption of rice increases as the moisture content increases. When the moisture content is high, the increase in water absorption of rice gradually stabilizes. There are three obvious steps for rice seeds to absorb water: First, at the beginning of water absorption, the water content of the seeds gradually increases, and the water absorption rate slowly increases. Second, during the peak water absorption period, the water absorption rate increases rapidly. Third, in the later stage of water absorption, the water content of seeds slowly increases, and the water absorption rate increases. The water absorption rate of rice seeds is closely related to temperature. In general, water absorption increases as temperature increases. The relationship between water absorption and temperature can be expressed as: unsaturated water absorption (%) = 14.289T-10.719 (where T is temperature, ℃)
```

# Output:

## Statement's scope:
```characteristics of soaking rice seeds to absorb sufficient water```

## Statement's main aspects and corresponding descriptions:
```
1. The water absorption curve
    1.1. The water absorption curve of rice after soaking is a unimodal curve.
    1.2. The inflection point is the peak period of water absorption.
    1.3. The relationship between rice water absorption and time is non-linear, which can be expressed by a*t+b=c, where a, b, and c are constants.
2. Rice water absorption as time changes
    2.1. Under different humidity conditions, the change in the water absorption rate of rice with time is basically similar.
    2.2. The water absorption rate is between 0 and the point d inflection.
    2.3. The rate of change accelerates over time.
    2.4. After the inflection point, the increase in water absorption gradually stabilizes.
3. Relationship between rice water absorption and moisture content
    3.1. The changing rules of water absorption and moisture content of rice are similar
    3.2. Under different humidity conditions, the relationship between water absorption of rice and moisture content is different.
    3.3. When the moisture content is low, the water absorption of rice increases as the moisture content increases.
    3.4. When the moisture content is high, the increase in water absorption of rice gradually stabilizes.
4. Three stages for rice seeds to absorb water
    4.1. At the beginning of water absorption, the water content of the seeds gradually increases, and the water absorption rate slowly increases.
    4.2. During the peak water absorption period, the water absorption rate increases rapidly.
    4.3. In the later stage of water absorption, the water content of seeds slowly increases, and the water absorption rate increases.
5. Relationship between water absorption rate and temperature
    5.1. The water absorption rate of rice seeds is closely related to temperature.
    5.2. Water absorption generally increases as temperature increases.
    5.3. The relationship can be expressed as: unsaturated water absorption (%) = 14.289T-10.719 (where T is temperature, ℃)
```


Now summarize the "scope" of the following statement with a few words, and then rephrase the input statement to its "main aspects" and "corresponding descriptions" in the numerically ordered two-layer format strictly. 
Note that the aspect list should be precise and concise to conclude the statement, and the number of aspects should be limited.
Each description sentence must be completed and faithful to the original statement, and you shoud NOT remove any descriptive segment in this layer.

# Input: 
```
{statement}
```

# Output: 
