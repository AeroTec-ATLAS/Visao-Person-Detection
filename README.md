Os ficheiros mais importantes aqui sao a accuracy_pipeline e a speed_pipeline. A minha ideia foi ter um buffer produtor-consumidor, onde uma thread frame_graber (produtor)
vai buscar frames e mete-os numa fila  e a main thread (consumidor) vai buscar os frames ao buffer. Se estivermos com velocidade suficiente (pouco lag), e nos pudermos 
focar em ter todos os frames, usamos a accuracy. 

Caso contrario, usamos a speed. O que a speed faz de diferente e apenas descartar frames que sejam mais antigos do que um dado threshold, para ir buscar frames mais
recentes.

o config.json tem as configuracoes de coisas importantes. E mais facil mexer nele do que se estivessem hard-coded

O ficheiro pipeline.py e uma versao simples que nao usa threads.

todos os outros (detect_people, rtsp, etc sao ficheiros velhos que fui deixando)


just for shits and giggles...

mandei ao chat as 3 pipelines e pedi lhe uma "unified_pipeline" - eu acho que ta meio merda mas fica ali 
