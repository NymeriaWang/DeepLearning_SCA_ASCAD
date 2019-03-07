clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matlab labeling SCA traces with SBOX(PxorK) %
%                                             %
% 2019, Dr.HuanyuWang                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% declaration of the SBOX 
SBOX=[099 124 119 123 242 107 111 197 048 001 103 043 254 215 171 118 ...
      202 130 201 125 250 089 071 240 173 212 162 175 156 164 114 192 ...
      183 253 147 038 054 063 247 204 052 165 229 241 113 216 049 021 ...
      004 199 035 195 024 150 005 154 007 018 128 226 235 039 178 117 ...
      009 131 044 026 027 110 090 160 082 059 214 179 041 227 047 132 ...
      083 209 000 237 032 252 177 091 106 203 190 057 074 076 088 207 ...
      208 239 170 251 067 077 051 133 069 249 002 127 080 060 159 168 ...
      081 163 064 143 146 157 056 245 188 182 218 033 016 255 243 210 ...
      205 012 019 236 095 151 068 023 196 167 126 061 100 093 025 115 ...
      096 129 079 220 034 042 144 136 070 238 184 020 222 094 011 219 ...
      224 050 058 010 073 006 036 092 194 211 172 098 145 149 228 121 ...
      231 200 055 109 141 213 078 169 108 086 244 234 101 122 174 008 ...
      186 120 037 046 028 166 180 198 232 221 116 031 075 189 139 138 ...
      112 062 181 102 072 003 246 014 097 053 087 185 134 193 029 158 ...
      225 248 152 017 105 217 142 148 155 030 135 233 206 085 040 223 ...
      140 161 137 013 191 230 066 104 065 153 045 015 176 084 187 022];

% Load the data 
keystru = load('keys.mat');
plaintextstru = load('plaintexts.mat');
tracestru = load('traces.mat');


keys=keystru.keys;
plaintexts=plaintextstru.plaintexts;
traces=tracestru.traces;

% Making testing set
TestKeys=keys(50001:51000,:);
TestPlaintexts=plaintexts(50001:51000,:);
TestTraces=traces(50001:51000,:);

% Making training set
TrainKeys=keys(1:50000,:);
TrainPlaintexts=plaintexts(1:50000,:);
TrainTraces=traces(1:50000,:);

% Make the label

KxorM=bitxor(keys,plaintexts);
labels =SBOX(KxorM+1);   %since the matrix from 0, but the data from 1.
label_3=labels(:,3);


% Making testing label
TestLabels=label_3(50001:51000,:);

% Making training set
TrainLabels=label_3(1:50000,:);


% Save all processed data
save('TrainLabels.mat','TrainLabels');
save('TrainTraces.mat','TrainTraces');
save('TestTraces.mat','TestTraces');
save('TestLabels.mat','TestLabels');