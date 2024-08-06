
set_sig_g=  .50;
max_sig=    .66;
delta=      .66;
max_sig_g=  .66;
min_sig_g= .025;
p_m_phi =   0;  % as the phi true is 0
min_ret = -0.5;max_ret = .75;
max_phi=0.05;min_phi=-0.05;   % change this if phi is believed larger than this range. This is specifically for zero phi
max_bta=p_m_bta+1.5;min_bta=p_m_bta-0.5;
mpme= 0;

%% The larger set_sig_g, the flatter the prior is, but we cannot set this number too big, it may make the chain cannot get out of trap state.
%B=set_sig_g;A=2*log(2)*B^2/(2*B+1); % This is only for Gamma

%% Uninformative Priors
p_std_alpha =   cstd;                                                                                                         % Prior for bta
p_std_phi =     cstd;
p_std_bta =     cstd*eye(Num_Factors);

%% Some more parameter settings                                                
g_ret_vec=[];
f_ret_vec=[];
p_v5=min_sig_g^2;

%% Going through each fund to see beginning and end of each one of them
T_start = [];T_end = [];
for i = 1:Num_Funds
  thisCF = CF_mat(:,i); % CF's for this fund
  ind = find(thisCF);
  T_start = [T_start ind(1)];
  T_end = [T_end ind(end)];
end

%% Initialize
acceptgam_vec = zeros(1,Num_Funds);                                  % Acceptance probs 
acceptalpha = 0;acceptbta = 0;acceptphi = 0;acceptsig_g = 0;acceptg_ret = zeros(T,1);                                            
E_f_ret = zeros(T,1);                                                % Vector to hold mean of simulated f_ret
std_f_ret = zeros(T,1);                                              % Vector to hold stdev of simulated f_ret
E_g_ret = zeros(T,1);                                                % Vector to hold mean of simulated f_ret
std_g_ret = zeros(T,1); 
PME_arr=[];
mpme_arr=[];
theta_mat = [];                                                      % Matrix to hold parameters

%% Initialize parameters
alpha = p_m_alpha;
phi =   p_m_phi;
bta=p_m_bta;
g_ret = alpha+Factors*p_m_bta+randn(T,1)*min_sig_g;   % Excess Return of Funds (return less risk-free rate)
