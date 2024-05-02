The API reference section of your documentation is a good place to include thorough, informational details on how to
use your code. A good way to do this is to include high-level diagrams on this page, and then have a separate page for 
each core module of your package. You can let `mkdocstrings` do most of the work for you on the module pages by pulling
in all your excellent docstrings directly from your code and formatting it nicely on your mkdocs webpage. Here is
a link on [how to do this](https://mkdocstrings.github.io/usage/).

Here is an example of displaying all the docstrings from the `example.py` module:

::: pdm_template_uq.example

And just for fun, here is an example UML class diagram using `Mermaid`.

``` mermaid
classDiagram
    class MyClass {
      +list[OtherClass] components
      +int another_property
      +my_method()
    }
    class OtherClass {
      +Array indices
      +activate_index(idx)
    }
    MyClass o-- "1..n" OtherClass
```