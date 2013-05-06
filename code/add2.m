function [y] = add2(varargin)
    disp(varargin)
    tmp=0;
    for i=1:length(varargin)
        tmp = tmp + varargin{i}
        
    end