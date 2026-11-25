%% PHM Society LaTeX Class
%% Based on standard LaTeX article class
%% Created by Jose Celaya, Indranil Roychoudhury, Matthew Daigle
%%
%% Revisions:
%% 	2012/01/15, created
%%  2015/03/23, revised: removed PHMbibliography, forced letterpaper, fixing widow/orphan issues, general cleanip
%%  2016/03/20, changed date

% Class setup
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{PHMSociety}[2016/03/20 v1.0 PHMSociety LaTeX Class]
\LoadClass[letterpaper]{article}

% ------------------------------------------------------------------------------------------
%
%: PACKAGE REQUIREMENTS
%
% ------------------------------------------------------------------------------------------
\RequirePackage{times}						% font selection
\RequirePackage{fancyhdr}					% for headers/footers
\RequirePackage{flushend}					% evening out final page
\RequirePackage[labelsep=period, format=plain]{caption}	% for captions
\RequirePackage{wrapfig}					% for biography photos
\RequirePackage[hang,flushmargin]{footmisc} % for footnotes
\RequirePackage{apacite}					% for citation style
\RequirePackage[english]{babel}		% for apacite setup
\RequirePackage{setspace}					% to enforce single spacing of bibliography

% ------------------------------------------------------------------------------------------
%
%: PAGE LAYOUT
%
% ------------------------------------------------------------------------------------------
\twocolumn  								% two column paper
\flushbottom 								% makes all text pages the same height, adding extra vertical space when necessary to fill out the page

% Text body size
\setlength\textheight{9.0in}
\setlength\textwidth{7.0in}

% Margins and column separation
% NOTE: The side margins assume that the margin origin is at 1" (\loffset), 
% 		So to make it 0.75", it needs to be set to negative.
% Check for reference: http://help-csli.stanford.edu/tex/latex-margins.shtml
\setlength\evensidemargin{-0.25in} % needed only if "two side" document version is selected, which should not be
\setlength\oddsidemargin{-0.25in}
\setlength\columnsep{0.25in}

% Header and top margins
% NOTE: The top margins assume that the origin is at 1" (\voffset) to the top of the header box
% Check for reference: http://help-csli.stanford.edu/tex/latex-margins.shtml
\setlength\topmargin{-0.5in}  % Move up to accommodate header box
\setlength\headheight{0.15in} % Header box is 0.5in hight
\setlength\headsep{0.35in}

% Title box setup
% TODO: For lots of authors, these settings may not work well. Consider modifying to accommodate, so use does not need to redefine titlebox.
\newlength\titlebox           % define new length variable
\setlength\titlebox{3.5in}    % This is a default value only, it might need to be modified in the *.tex file to 
                              % account for cases with more affiliations and authors that will require a 
                              % bigger box

% Widow/orphan control (reduce bad boxes)
\tolerance 1414
\hbadness 1414
\emergencystretch 1.5em
\hfuzz 0.3pt
\widowpenalty=10000
\vfuzz \hfuzz
% NOTE: typically raggedbottom is suggested here, but we prefer flushbottom.


% ------------------------------------------------------------------------------------------
%
%: HEADER AND FOOTER CONTENT DECLARATIONS
%
% ------------------------------------------------------------------------------------------

% No header/footer for first page
\thispagestyle{empty}
\pagestyle{fancy}
\fancyhead{}
\fancyfoot{}

% Set up publication year, which may be given as an option
\def\publicationYear{20xx}

\DeclareOption{2013}{
\def\publicationYear{2013}
}

\DeclareOption{2014}{
\def\publicationYear{2014}
}

\DeclareOption{2015}{
\def\publicationYear{2015}
}

\DeclareOption{2016}{
\def\publicationYear{2016}
}

\DeclareOption{2017}{
\def\publicationYear{2017}
}

\DeclareOption{2018}{
\def\publicationYear{2018}
}

\DeclareOption{2019}{
\def\publicationYear{2019}
}

\DeclareOption{2020}{
\def\publicationYear{2020}
}

% TODO: For IJPHM options, footer should include the paper number

% set up correct header based on publication
\DeclareOption{IJPHM}{%
\fancyhead[C]{\small{\textsc{International Journal of Prognostics and Health Management}}}%
\renewcommand{\headrulewidth}{0pt}%					% Remove head ruler
\fancyfoot[R]{{\small{\thepage}}}% --- footer all pages
\fancypagestyle{plain}{% --- first page header and footer content
   \fancyhead[C]{}}% --- Clear first page header
\fancyfoot[L]{\small{International Journal of Prognostics and Health Management, ISSN2153-2648, \publicationYear }}%	% --- Write first page footer (Journal Only)
}

\DeclareOption{IJPHM-TB}{%
\fancyhead[C]{\small{\textsc{Technical Brief}}}%
\renewcommand{\headrulewidth}{0pt}%					% Remove head ruler
\fancyfoot[R]{{\small{\thepage}}}% --- footer all pages
\fancypagestyle{plain}{% --- first page header and footer content
   \fancyhead[C]{}}% --- Clear first page header
\fancyfoot[L]{\small{International Journal of Prognostics and Health Management, ISSN2153-2648, \publicationYear}}%	% --- Write first page footer (Journal Only)
}

\DeclareOption{IJPHM-COMM}{%
\fancyhead[C]{\small{\textsc{Communication}}}%
\renewcommand{\headrulewidth}{0pt}%					% Remove head ruler
\fancyfoot[R]{{\small{\thepage}}}% --- footer all pages
\fancypagestyle{plain}{% --- first page header and footer content
   \fancyhead[C]{}}% --- Clear first page header
\fancyfoot[L]{\small{International Journal of Prognostics and Health Management, ISSN2153-2648, \publicationYear }}%	% --- Write first page footer (Journal Only)
}

\DeclareOption{PHME}{%
\fancyhead[C]{\small{\textsc{European Conference of the Prognostics and Health Management Society \publicationYear}}}% \the\year}}}%
\renewcommand{\headrulewidth}{0pt}%					% Remove head ruler
\fancyfoot[R]{{\small{\thepage}}}% --- footer all pages
\fancypagestyle{plain}{% --- first page header and footer content
   \fancyhead[C]{}}% --- Clear first page header
}

\DeclareOption{PHM}{%
\fancyhead[C]{\small{\textsc{Annual Conference of the Prognostics and Health Management Society \publicationYear}}}%
\renewcommand{\headrulewidth}{0pt}%					% Remove head ruler
\fancyfoot[R]{{\small{\thepage}}}% --- footer all pages
\fancypagestyle{plain}{% --- first page header and footer content
   \fancyhead[C]{}}% --- Clear first page header
}

\ProcessOptions\relax

% ------------------------------------------------------------------------------------------
%
%: FONT SIZE DECLARATIONS
%
% ------------------------------------------------------------------------------------------

% See the following for more info:
% - http://wiki.lyx.org/FAQ/ChangeFontUsingLatex
% - http://xpt.sourceforge.net/techdocs/language/latex/latex32-LaTeXAndFonts/ar01s14.html
% These are different than default sizes, so we must specify them here
\def\@normalsize{\@setsize\normalsize{10pt}\xpt\@xpt}         % 10 point on 11
\def\normalsize{\@setsize\normalsize{10pt}\xpt\@xpt}          % 10 point on 11
\def\small{\@setsize\small{9pt}\ixpt\@ixpt}                   % 9 point on 10
\def\footnotesize{\@setsize\footnotesize{8pt}\viiipt\@viiipt} % 9 point on 10
\def\scriptsize{\@setsize\scriptsize{8pt}\viiipt\@viiipt}     % 7 point on 8
\def\tiny{\@setsize\tiny{7pt}\viipt\@viipt}                   % 6 point on 7
\def\large{\@setsize\large{12pt}\xiipt\@xiipt}                % 11 point on 12
\def\Large{\@setsize\Large{14pt}\xivpt\@xivpt}                % 12 point on 14
\def\LARGE{\@setsize\LARGE{16pt}\xviipt\@xviipt}              % 14 point on 16
\def\huge{\@setsize\huge{18pt}\xviiipt\@xviiipt}          		% 17 point on 20
\def\Huge{\@setsize\Huge{23pt}\xxiiipt\@xxiiipt}              % 20 point on 23
\def\email{\@setsize\email{9pt}\ixpt\@ixpt}              			% 9 point

% ------------------------------------------------------------------------------------------
%
%: PARAGRAPH DECLARATIONS
%
% ------------------------------------------------------------------------------------------

% Line spacing
% NOTE: Use \linespread{1.3} for "one and a half" line spacing, and \linespread{1.6} for "double" line spacing.
\linespread{1.2}

% Indentation
\setlength{\parindent}{0pt}

% Spacing between paragraphs
\setlength{\parskip}{6pt}
% NOTE: \parskip will affect lists too. This needs to be fixed later on the lists environments.

% ------------------------------------------------------------------------------------------
%
%: TITLE BLOCK DECLARATIONS
%
% ------------------------------------------------------------------------------------------

\def\@maketitle{%
	\def\and{\unskip{, }}
  	\def\And{\unskip{, and }}
	\vskip 0.55in                     % Space before title is 0.5"
	\begin{center}
		{\LARGE\bf\@title\par}          % Title text
		\vskip 14pt                     % Space after title, 14pt
		{\normalsize \@author \par}     % Author text			
		\vskip 0.02in                   % Space after authors, 12pt
		{\normalsize\it                 % Affiliation
		\vskip -1pt
		\begin{tabular}[t]{c}				
			\address
		\end{tabular}
		\par}
		%
		\vskip 26pt                     % Space after affiliation, 24 + 3 pt
		%
	\end{center}
}% End maketitle definition

% ------------------------------------------------------------------------------------------
%
%: ABSTRACT ENVIRONMENT DECLARATIONS
%
% ------------------------------------------------------------------------------------------

% Renew abstract environment to have centered bold small caps title
\renewenvironment{abstract}{\centerline{\normalsize\bfseries\scshape Abstract}}

% ------------------------------------------------------------------------------------------
%
%: LICENSE FOOTNOTE DECLARATIONS
%
% ------------------------------------------------------------------------------------------

% Define phmLicenseFootnote command
\newcommand{\phmLicenseFootnote}[1]{%
\insert\footins{\noindent\footnotesize{#1~et al. This is an open-access article distributed under the terms of the Creative Commons Attribution 3.0 United States License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.}}
}

% ------------------------------------------------------------------------------------------
%
%: AFFILIATIONS DECLARATIONS
%
% ------------------------------------------------------------------------------------------

\newcommand{\authorNumber}[1]{\textnormal{\textsuperscript{#1}}}  % This command is for formatting the author 
                                                                  % number as a subscript
\newcommand{\affiliation}[3]{\\ \authorNumber{#1} #2 \\ #3}
\newcommand{\address}[1]{\def\address{#1}}

% ------------------------------------------------------------------------------------------
%
%: SECTIONS DECLARATIONS
%
% ------------------------------------------------------------------------------------------

% Section font style and before/after spacing   
\def\section{\@startsection{section}{1}{\z@}{6pt}{0.01pt}{\normalsize\bf\scshape}}
\def\subsection{\@startsection{subsection}{2}{\z@}{6pt}{0.01pt}{\normalsize\bf}}
\def\subsubsection{\@startsection{subsubsection}{3}{\z@}{6pt}{0.01pt}{\normalsize\bf}}
% See notes on Latex sections here: http://help-csli.stanford.edu/tex/latex-sections.shtml

% Define section numbering styles
\def\thesection{\arabic{section}}
\def\thesubsection{\thesection.\arabic{subsection}}
\def\thesubsubsection{\thesubsection.\arabic{subsubsection}}

% Redefine bibliography so that it uses single spacing
\newcommand*\origbib{}
\let\origbib\bibliography
\renewcommand*\bibliography{\singlespace \origbib}

% Update section formatting to include period after the section number, to match MS Word style
\renewcommand*{\@seccntformat}[1]{
       \csname the#1\endcsname.
}
% The "\csname the#1\endcsname" part puts in the section number, then the 
% "." there puts in the ".", and then the "\quad" inserts some space. 
% There is a description here: http://www.tex.ac.uk/cgi-bin/texfaq2html?label=seccntfmt

% ------------------------------------------------------------------------------------------
%
%: FLOAT SPACING DECLARATIONS
%
% ------------------------------------------------------------------------------------------

% Set these to ensure nice spacing with floats and other display enviornments
\floatsep 12pt plus 2pt minus 2pt
\textfloatsep 16pt plus 2pt minus 4pt
\intextsep 12pt plus 2pt minus 2pt
\dblfloatsep 12pt plus 2pt minus 2pt
\dbltextfloatsep 18pt plus 2pt minus 4pt
\abovedisplayskip 7pt plus2pt minus5pt%
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip  0pt plus3pt%   
\belowdisplayshortskip  4pt plus3pt minus3pt%

% ------------------------------------------------------------------------------------------
%
%: LISTS DECLARATIONS
%
% ------------------------------------------------------------------------------------------

% Indentation for all the lists and sublists. Applies to enumerate, itemize
\leftmargini 0.25in    % This should be 0.25in which is the column separation size
\leftmarginii 2em
\leftmarginiii 1em 
\leftmarginiv 0.5em
\leftmarginv 0.5em 
\leftmarginvi 0.5em

% Applies to enumerate, itemize and description
\leftmargin\leftmargini
\labelsep 10pt             % separation from label to text (update w/MSWord dim.)
\labelwidth\leftmargini\advance\labelwidth-\labelsep

% Applies to enumerate, itemize and description
\def\@listI{\leftmargin\leftmargini 
\parsep 0pt plus 1pt minus 0.5pt           % This is the element to element separation
\topsep 0pt plus 1pt minus 2pt             % This is the top separation
\itemsep 3pt plus 1pt minus 0.5pt          % This is the element to element separation (repeated)
\partopsep 6pt plus 0.5pt minus 0.5pt}     % Bottom separation

\let\@listi\@listI
\@listi

\def\@listii{\leftmargin\leftmarginii
   \labelwidth\leftmarginii\advance\labelwidth-\labelsep
   \parsep 1pt plus 0.5pt minus 0.5pt
   \topsep 2pt plus 1pt minus 0.5pt
   \itemsep \parsep}
\def\@listiii{\leftmargin\leftmarginiii
    \labelwidth\leftmarginiii\advance\labelwidth-\labelsep
    \parsep 0pt plus 1pt
    \partopsep 0.5pt plus 0pt minus 0.5pt
    \topsep 1pt plus 0.5pt minus 0.5pt 
    \itemsep \topsep}
\def\@listiv{\leftmargin\leftmarginiv
     \labelwidth\leftmarginiv\advance\labelwidth-\labelsep}
\def\@listv{\leftmargin\leftmarginv
     \labelwidth\leftmarginv\advance\labelwidth-\labelsep}
\def\@listvi{\leftmargin\leftmarginvi
     \labelwidth\leftmarginvi\advance\labelwidth-\labelsep}
