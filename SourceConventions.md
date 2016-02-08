# Source code formatting conventions

[TOC]

The following conventions apply to all code written in HOOMD-blue, whether it be C++, python, CMake, etc. Pull requests
will not be accepted and feature branches will not be merged unless these conventions are followed.

## Indentation

* *Spaces* are used to indent lines. (see editor configuration guide below)
* A single level of indentation is to be *4 spaces*
* There should be no whitespace at the end of lines in the file.
* The indentation style used is [Whitesmith's style](http://en.wikipedia.org/wiki/Indent_style#Whitesmiths_style). An extended set of examples follows:

```
#!c++
class SomeClass
    {
    public:
        SomeClass();
        int SomeMethod(int a);
    private:
        int m_some_member;
    };

// indent function bodies
int SomeClass::SomeMethod(int a)
    {
    // indent loop bodies
    while (condition)
        {
        b = a + 1;
        c = b - 2;
        }

    // indent switch bodies and the statements inside each case
    switch (b)
        {
        case 0:
            c = 1;
            break;
        case 1:
            c = 2;
            break;
        default:
            c = 3;
            break;
        }

    // indent the bodies of if statements
    if (something)
        {
        c = 5;
        b = 10;
        }
     else if (something_else)
        {
        c = 10;
        b = 5;
        }
     else
        {
        c = 20;
        b = 6;
        }

    // omitting the braces is fine if there is only one statement in a body (for loops, if, etc.)
    for (int i = 0; i < 10; i++)
        c = c + 1;

    return c;
    // the nice thing about this style is that every brace lines up perfectly with it's mate
    }

```

* Documentation comments and items broken over multiple lines should be *aligned* with spaces

```
#!c++
class SomeClass
    {
    private:
        int m_some_member;        //!< Documentation for some_member
        int m_some_other_member;  //!< Documentation for some_other_member
    };

template<class BlahBlah> void some_long_func(BlahBlah with_a_really_long_argument_list,
                                             int b,
                                             int c);
```

## Formatting Long Lines

* All code lines should be hand-wrapped so that they are no more than *120 characters* long
* Simply break any excessively long line of code at any natural breaking point to continue on the next line


```
#!c++
cout << "This is a really long message, with "
     << message.length()
     << "Characters in it:"
     << message << endl;
```

* Try to maintain some element of beautiful symmetry in the way the line is broken. For example, the _above_ long message is preferred over the below:

```
#!c++
cout << "This is a really long message, with " << message.length() << "Characters in it:"
   << message << endl;
```

* There are *special rules* for function definitions and/or calls
* If the function definition (or call) cleanly fits within the 120 character limit, leave it all on one line
```
#!c++
int some_function(int arg1, int arg2)
```
* (option 1) If the function definition (or call) goes over the limit, you may be able to fix it by simply putting the template definition on the previous line:
```
#!c++
// go from
template<class Foo, class Bar> int some_really_long_function_name(int with_really_long, Foo argument, Bar lists)
// to
template<class Foo, class Bar>
int some_really_long_function_name(int with_really_long, Foo argument, Bar lists)
```
* (option 2) If the function doesn't have a template specifier, or splitting at that point isn't enough, split out each argument onto a separate line and align them.
```
#!c++
// go from
int some_really_long_function_name(int with_really_long_arguments, int or, int maybe, float there, char are, int just, float a, int lot, char of, int them)
// to
int some_really_long_function_name(int with_really_long_arguments,
                                   int or,
                                   int maybe,
                                   float there,
                                   char are,
                                   int just,
                                   float a,
                                   int lot,
                                   char of,
                                   int them)
```

## Documentation Comments

* Every class, member variable, function, function parameter, macro, etc. *MUST* be documented with *doxygen* comments.
* See http://www.stack.nl/~dimitri/doxygen/docblocks.html
* If you copy an existing file as a template, *DO NOT* simply leave the existing documentation comments there. They apply to the original file, not your new one!
* The best advice that can be given is to write the documentation comments *FIRST* and the _actual code_ *second*. This allows one to formulate their thoughts and write out in English what the code is going to be doing. After thinking through that, writing the actual code is often _much easier_, plus the documentation left for future developers to read is top-notch.
* Good documentation comments are best demonstrated with an in-code example. See in particular the long section in TablePotential.h where the all the details about the calculation and data layout were specified before even a single line of code was written.

# Configuring Git

You can configure your git clone to reject commits that make errors that use tabs for indentation, or add whitespace to the end of file lines. First, set your options.
```
#!c++
git config --global core.whitespace trailing-space,space-before-tab,tab-in-indent,tabwidth=4
```
Then, go to the hooks directory and enable the sample pre-commit hook.
```
#!c++
$ cd hoomd-blue
$ cd .git/hooks/
$ mv pre-commit.sample pre-commit
```
Now, any commit that uses tabs for indentation or adds whitespace at the end of lines will fail. The failure message will indicate which files have the problems (tools like gitx can highlight the problem areas).

Unfortunately, these settings do not transfer with clones and the hook setup step must be performed on each clone you make. The `git config --global` line only needs to be executed once on each computer.

# Configuring Editors

The following editor configuration settings are recommended when working on HOOMD-Blue files. They provide a minimal set of options to:
* Set the indentation to 4 spaces
* Make dealing with indentation spaces as if they were tabs (i.e., a single backspace will delete 4 spaces of indentation)
* Provide a simple auto-indent that makes working with Whitesmith's style easy.
* Provide a rule at column 120 so it is easy to see when long lines go over the limit

## Sublime Text

Sublime text configuration is included with HOOMD-blue. Simply open the project `hoomd.sublime-project` in Sublime text. Command-B is linked to build an already configured build dir in `build/`

The following packages make HOOMD development easier (install using Package Control).

* CMake - syntax highlighting for cmake files
* Git - git commands from within ST
* TrailingSpaces - highlights and removes trailing whitespace from lines

## Vim

Recommended settings for your .vimrc file

```
#!vim
" set tab and indent size to 4 real spaces "
:set ts=4
:set sw=4
:set expandtab

" set soft tab settings so backspace intelligently deletes 4 spaces as if they were a tab "
:set sts=4
:set backspace=indent,eol,start

" enable simple autoindenting "
:set ai

" options to make the UI more friendly "
:set ruler
:syntax enable
:set nobackup

" Highlight unwanted spaces/tags in red "
:highlight ExtraWhitespace ctermbg=red guibg=red
:match ExtraWhitespace /\s\+$\|\t/
```

## Emacs

The cc-mode that is used by default in Emacs and XEmacs for c/cpp code has a predefined style for Whitesmith-formatting. Unfortunately, that definition does not fully comply with the coding style documented here. So one needs to define a *hoomd*-style. The best way to include customizations into cc-mode is via adding instructions to the c-mode-common-hook in your initialization file, the name of which depends on the kind and version of (X)Emacs. Here is some example that has been tested with XEmacs 21.5 and cc-mode 5.30.10 that only changes setting for c/cpp/cuda files in the HOOMD-blue tree:


```
#!emacs
;; define hoomd-indentation style. this is a modified version of the "whitesmith" style.
(defconst my-hoomd-style
  '((c-tab-always-indent        . t)
    (c-basic-offset . 4)
    (c-comment-only-line-offset . 0)
    (c-offsets-alist
     (knr-argdecl-intro . +)
     (label . 0)
     (statement-cont . +)
     (substatement-open . +)
     (substatement-label . +)
     (block-open . +)
     (statement-block-intro . c-lineup-whitesmith-in-block)
     (block-close . c-lineup-whitesmith-in-block)
     (inline-open . +)
     (defun-open . +)
     (defun-block-intro . c-lineup-whitesmith-in-block)
     (defun-close . c-lineup-whitesmith-in-block)
     (brace-list-open . +)
     (brace-list-intro . c-lineup-whitesmith-in-block)
     (brace-entry-open . c-indent-multi-line-block)
     (brace-list-close . c-lineup-whitesmith-in-block)
     (class-open . +)
     (inclass . +)
     (class-close . +)
     (inexpr-class . +)
     (access-label . -)
     (extern-lang-open . +)
     (inextern-lang . c-lineup-whitesmith-in-block)
     (extern-lang-close . +)
     (namespace-open . +)
     (innamespace . c-lineup-whitesmith-in-block)
     (namespace-close . +)
     (module-open . +)
     (inmodule . c-lineup-whitesmith-in-block)
     (module-close . +)
     (composition-open . +)
     (incomposition . c-lineup-whitesmith-in-block)
     (composition-close . +)
     )
    (c-echo-syntactic-information-p . t))
    "HOOMD-blue Programming Style")

(require 'cc-mode)
(c-add-style "hoomd" my-hoomd-style)

;; define customization function
(defun my-c-mode-common-hook ()
  "Customized settings for cc-mode"
  (message "activating personal cc-mode customization")
  (if (string-match "./hoomd/.*/.*\\.\\(h\\|cc\\|cu\\|cuh\\)$" (buffer-file-name))
     (progn (message "Detected HOOMD-blue tree. Turning on HOOMD-blue formatting settings.")
       (c-set-style "hoomd")
       (setq c-ignore-auto-fill '(string cpp))
       (auto-fill-mode 1)
       (setq fill-column 120)
       (setq tab-width 4)
       (setq indent-tabs-mode nil)))
)
(add-hook 'c-mode-common-hook 'my-c-mode-common-hook)

;; recognize HOOMD/CUDA related file type extensions
(setq auto-mode-alist
      (append '(("\\.cc$"   . cpp-mode)
                ("\\.cu$"   . cpp-mode)
                ("\\.cuh$"  . cpp-mode)
                ("\\.h$"    . cpp-mode)
                ("\\.hoomd$"  . python-mode)
                ("\\.vmd$"  . tcl-mode))
              auto-mode-alist))
```

## (Your Favorite Editor Here)

Edit this document and add the settings for use in your favorite editor.
