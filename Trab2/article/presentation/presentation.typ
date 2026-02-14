#import "@preview/fletcher:0.5.0" as fletcher: diagram, node, edge, shapes
#import "@preview/touying:0.5.2": *


#let ufesblue = rgb("#009fe3")
#let ufesdarkblue = rgb("#174578")

#show footnote.entry: set text(size: 14pt)

// https://touying-typ.github.io/docs/themes/metropolis
#import themes.metropolis: *
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.institution,
  config-info(
    title: [Otimizando o Desempenho de Rede Neural em Jogos usando Simulated Annealing e Apredizagem por Reforço],
    // subtitle: [Subtitle],
    author: [
      Henrique Coutinho Layber#footnote[henrique.layber\@edu.ufes.br]\
      Vitor Berger Bonella#footnote[vitor.bonella\@edu.ufes.br]\
      Flávio Miguel Varejão#footnote[flavio.varejao\@ufes.br]
    ],
    date: [Outubro 2024],
    institution: [
      Universidade Federal do Espírito Santo\
      // Departamento de Informática // Shows on every footer, so omit
    ],
    // Bugs out on the first page
    // logo: image("ufes_notext.svg")
  ),
  config-colors(secondary: ufesdarkblue),
)

// Special first page to replicate UFES template
#slide(
  config: config-page(margin: 0pt, footer: none),
)[
  #grid(
    rows: (4fr, 3fr),
    columns: 1fr,
    align: horizon + center,

    image("Marca_Ufes_SVG.svg", height: 100%),
    grid.cell(fill: ufesblue)[
      #set text(weight: "bold", fill: navy)
      #pad(30pt)[
        Optimizing Neural Network Performance in Game Playing Using Simulated Annealing and Reinforcement Learning
      ]
    ]
  )
]


#title-slide()

= Introdução

// Pretendemos experimentar isso e descobrir qual a melhor rotina de arrefecimento

= Método proposto

// Vamos fazer assim pra descobrir isso e aquilo

== Jogo Dino

// O jogo é assim e isso e aquilo e speed funciona assim e ponto assado
// O jogo roda em CPU somente, em uma reimplementação dele no PyGame.

== Simulated Annealing

// É assim que funciona e esses são os pontos bons

== Simulated Annealing::Rotinas de Arrefecimento

== Rede Neural

// Incluir diagrama usado no camera-ready

== Rede Neural::Aprendizagem por Reforço

// É assim que RL funciona


= First Section

== A long long long long long long long long long long long long long long long long long long long long long long long long Title

A slide with equation:

$ x_(n+1) = (x_n + a / x_n) / 2 $

#lorem(200)

= Second Section

#focus-slide[
  Wake up!
]

== Simple Animation

We can use `#pause` to #pause display something later.

#meanwhile

Meanwhile, #pause we can also use `#meanwhile` to display other content synchronously.

#speaker-note[
  + This is a speaker note.
  + You won't see it unless you use `config-common(show-notes-on-second-screen: right)`
]

#show: appendix

= Appendix

Please pay attention to the current slide number.
