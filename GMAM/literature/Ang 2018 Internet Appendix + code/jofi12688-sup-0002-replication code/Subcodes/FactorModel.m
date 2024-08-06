if wfactor==1
icol=[9];
end
if wfactor==3
icol=[9:11];
end
if wfactor==4
icol=[9:11 14];
end
if wfactor==5
icol=[scol];
end
if wfactor==6
icol=[scol 10 11];
end
if wfactor==7
icol=[scol 10:11 14];
end
if wfactor==11 % Fama French five factor: MKTRF SMB HML RMW CMA
    icol=[9:11 43 44];
end
if wfactor==12 % Carhart 4 factor + QMJ (AQR)
    icol=[9:11 30 45];
end
if wfactor==8 % Traded Fund 1 factor
    icol=[7];
end
if wfactor==9 % traded fund 3 factor
    icol=[7 5 6];
end
if wfactor==10 % traded fund 4 factor
    icol=[7 5 6 46];
end
Num_Factors=size(icol,2);
p_m_bta=p_m_bta(1,1:Num_Factors)';
Factors=FactorM(FactBeg:FactEnd,icol);