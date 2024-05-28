import React from 'react'; 
import { Menubar } from 'primereact/menubar';
import "primereact/resources/themes/lara-light-cyan/theme.css";
import 'primeicons/primeicons.css';

export default function NavBar() {

    const items = [
        {
            label: 'Home',
            icon: 'pi pi-home',
            command:()=>{ window.location="Home"; }
        },
        {
            label: 'Monitor',
            icon: 'pi pi-star',
            command:()=>{ window.location="Monitor"; }
        },
        {
            label: 'Projects',
            icon: 'pi pi-search',
            items: [
                {
                    label: 'Components',
                    icon: 'pi pi-bolt'
                },
                {
                    label: 'Blocks',
                    icon: 'pi pi-server'
                },
                {
                    label: 'UI Kit',
                    icon: 'pi pi-pencil'
                },
                {
                    label: 'Templates',
                    icon: 'pi pi-palette',
                    items: [
                        {
                            label: 'Apollo',
                            icon: 'pi pi-palette'
                        },
                        {
                            label: 'Ultima',
                            icon: 'pi pi-palette'
                        }
                    ]
                }
            ]
        },
        {
            label: 'Contact',
            icon: 'pi pi-envelope'
        }
    ];


    return (
        <div className="card">
            <Menubar model={items} />
        </div>
    )
}
        