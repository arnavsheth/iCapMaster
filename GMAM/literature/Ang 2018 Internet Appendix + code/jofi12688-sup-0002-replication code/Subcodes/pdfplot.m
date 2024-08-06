function pdfplot(xin,sig, nbins)
% pdfplot(X, nbins) 
%   displays a histogram of the empirical probability density function (PDF) 
%   for the data in the input array X using nbins number of bins 
%   (by default pdfplot sets nbins to 20).
%   If input X is a matrix, then pdfplot(X) parses it to the vector and 
%   displays PDF of all values.
%   For complex input X, pdfplot(X) displays PDF of abs(X).
%
%   Example:
%    y = randn( 1, 1e5 );
%    pdfplot( y );
%    pdfplot( y, 100 );

% Version 1.0
% Alex Bur-Guy, September 2003
% alex@wavion.co.il
%
% Revisions:
%       Version 1.0 -   initial version

if nargin == 2, nbins = 20; end
xin = reshape( xin, numel(xin), 1 );
if ~isreal( xin ), xin = abs( xin ); end
minXin = min(xin); maxXin = max(xin);
if floor( nbins ) ~= nbins, error( 'Number of bins should be integer value' ); end
if nbins < 2, error( 'Number of bins should be positive integer greater than 1 ' ); end
if minXin == maxXin
     bar(minXin,1);
     axis([minXin - 10, minXin + 10, 0, 1]);
else
     step = (maxXin - minXin) / (nbins-1);
     binc = minXin : step : maxXin;     
     [N, X] = hist(xin, binc);
     bar(X, N/trapz(X,N));
end


hold on;


x=minXin:(maxXin-minXin)/100:maxXin;
plot(x,normpdf(x,0,sig))
hold off;

xlabel('log(PME)', 'FontWeight','b','FontSize',12);
%title(['PDF(X) based on ' num2str(length(xin)) ' data samples @ ' num2str(nbins) ' bins'], 'FontWeight','b','FontSize',12);