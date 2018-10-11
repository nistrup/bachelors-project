# Bachelor 2018 - CBS HA(mat.)

Teorien bygger på Harry Markowitz moderne portefølje teori.
Skrevet med udgangspunkt i DeMiguel, Garlappi og Uppal's artikkel "Optimal Versus Naive Diversification:
How Inefficient is the 1/N Portfolio Strategy?" - http://faculty.london.edu/avmiguel/DeMiguel-Garlappi-Uppal-RFS.pdf

## Modeller vi har programmeret:

```
Naive
  
  1. 1/N with rebalancing (benchmark strategy)                ew or 1/N
  
Classical approach that ignores estimation error
  
  2. Sample-based mean-variance                               mv
  
Bayesian approach to estimation error
  
  4. Bayes-Stein                                              bs
  
Moment restrictions
  
  6. Minimum-variance                                         min
  
Portfolio constraints
  
  9. Sample-based mean-variance with shortsale constraints    mv-c
  10. Bayes-Stein with shortsale constraints                  bs-c
  11. Minimum-variance with shortsale constraints             min-c
  12. Minimum-variance with generalized constraints           g-min- c
```

## Modeller vi mangler:

```
Bayesian approach to estimation error
  
  5. Bayesian Data-and- Model                                 dm

Moment restrictions
  
  7. Value-weighted market portfolio                          vw
  8. MacKinlay and Pastor’s (2000) missing-factor model       mp
  
Optimal combinations of portfolios
  
  13. Kan and Zhou’s (2007) “three-fund” model                mv-min
  14. Mixture of minimum-variance and 1/N                     ew-min
```

## Data vi har samlet:

```
#1 - Ten sector portfolios of the S&P 500 and the US equity market portfolio
#2 - Ten industry portfolios and the US equity market portfolio
#3 - Eight country indexes and the World Index
#4 - SMB and HML portfolios and the US equity market portfolio
#5 - Twenty size- and book-to-market portfolios and the US equity MKT
#6 - Twenty size- and book-to-market portfolios and the MKT, SMB, and HML portfolios
#7 - Twenty size- and book-to-market portfolios and the MKT, SMB, HML, and UMD portfolios
```

## Sammenligning med DeMiguel, Garlappi og Uppal's resultater:

![alt text](https://i.imgur.com/gnO3oT3.png)

# Rå resultater

## Sharpe-værdier:

![alt text](https://i.imgur.com/O8YxgcV.png)

## Maxima- og minima-vægtninger:

![alt text](https://i.imgur.com/fRMZ1dz.png)

## Turnover relativt til 1/N:

![alt text](https://i.imgur.com/ZDk9jLC.png)

## Sharpe-værdier fra simuleret data:

![alt text](https://i.imgur.com/bZ7ik9Q.png)

Bachelors 2018, Optimal Portfolio Theory using random portfolio simulations

Af Peter Nistrup og Christoffer Fløe
